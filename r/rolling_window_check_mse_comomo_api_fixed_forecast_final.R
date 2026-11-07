# Rolling-origin using only the CoMoMo package (metadata, stack, bma,
# mcs, frequentist, fitCoMoMo, CoMoMo, forecast). 
# Covers: stack (meta-learners from config: nnls, linear, lasso, ridge, elastic),
# bma (cv + sv), mcs (cv + sv), frequentist, and simple average via forecast(CoMoMo(., weight = NULL)).
# Stacked regression requires explicit stack_metalearners in the JSON (or MORTALITY_STACK_METALEARNERS); use [] for none.
#
# Evaluation (MSE): for each origin in rolling_origins, fit on expanding calendar years
# fit_start:origin (same as rolling_window_check_mse_comomo_api_fixed.R); forecast horizon
# per origin is min(maxYear - origin, H_MAX) so weights (h = 1..H_MAX) stay valid.
#
# True future forecast (run_mode forecast/both): one fit on all HMD years >= fit_start through
# the latest year in the downloaded StMoMo object; then forecast(..., h = forecast_horizon).
# Weights for that step are re-learned on years_fit_full (same span as fitModels_full), while
# rolling evaluation still uses metadata + weight_specs on the fixed config fit_period (yearsFit).
#
# CSVs: MORTALITY_OUTPUT_DIR or default output/comomo_new_fixed_forecast/

# ---- Packages: project .Rlib first; then CRAN deps; then CoMoMo (local tree preferred) ----
dir.create(".Rlib", showWarnings = FALSE, recursive = TRUE)
rlib <- normalizePath(".Rlib", mustWork = FALSE)
.libPaths(c(rlib, .libPaths()))

suppressPackageStartupMessages(
  invisible(lapply(
    c("jsonlite", "demography", "dplyr", "tidyr", "StMoMo", "forecast", "ggplot2"),
    library, character.only = TRUE
  ))
)

load_comomo <- function(path = "CoMoMo") {
  load_local <- function(pkg) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      return(FALSE)
    }
    getExportedValue(pkg, "load_all")(normalizePath(path, mustWork = TRUE), quiet = TRUE)
    TRUE
  }

  if (!dir.exists(path)) {
    if (!requireNamespace("CoMoMo", quietly = TRUE)) {
      stop("CoMoMo not found: Please install the package.")
    }
    suppressPackageStartupMessages(library(CoMoMo))
    return(invisible())
  }

  if (load_local("pkgload") || load_local("devtools")) {
    return(invisible())
  }

  message("CoMoMo: installing from CoMoMo/ into .Rlib (install pkgload or devtools to skip)...")
  utils::install.packages(
    normalizePath(path, mustWork = TRUE),
    lib = rlib,
    repos = NULL,
    type = "source",
    quiet = TRUE
  )
  suppressPackageStartupMessages(library(CoMoMo, lib.loc = rlib))
}
load_comomo()

# Optional .env: KEY=VALUE -> Sys.setenv (only if that key is not already set).
load_dotenv <- function(path = ".env") {
  if (!file.exists(path)) {
    return(invisible())
  }
  for (ln in readLines(path, warn = FALSE, encoding = "UTF-8")) {
    ln <- sub("^#.*$", "", trimws(ln))
    if (!nzchar(ln) || !grepl("=", ln, fixed = TRUE)) {
      next
    }
    key <- trimws(sub("=.*", "", ln, perl = FALSE))
    val <- trimws(sub("^[^=]+=", "", ln))
    val <- gsub("^[\"']|[\"']$", "", val)
    if (nzchar(key) && !nzchar(Sys.getenv(key, unset = ""))) {
      do.call(Sys.setenv, stats::setNames(list(val), key))
    }
  }
  invisible()
}
load_dotenv(".env")

# HMD login for hmd.mx(): accept USER/USERNAME and PASS/PASSWORD aliases.
hmd_env <- function(a, b) {
  u <- Sys.getenv(a, unset = "")
  if (nzchar(u)) {
    u
  } else {
    Sys.getenv(b, unset = "")
  }
}
hmd_user <- hmd_env("HMD_USERNAME", "HMD_USER")
hmd_pass <- hmd_env("HMD_PASSWORD", "HMD_PASS")
if (!nzchar(hmd_user) || !nzchar(hmd_pass)) {
  stop("Set HMD_USERNAME or HMD_USER and HMD_PASSWORD or HMD_PASS (e.g. in .env).")
}

# ---- JSON config: data subset, fit period, rolling range, horizons, models list ----
cfg_path <- Sys.getenv("MORTALITY_EFFECTIVE_CONFIG", unset = "effective_config.json")
if (!file.exists(cfg_path)) {
  cfg_path <- file.path("config", "baseline_effective.json")
}
if (!file.exists(cfg_path)) {
  stop("Missing config: effective_config.json or config/baseline_effective.json")
}
cfg <- jsonlite::fromJSON(cfg_path, simplifyVector = TRUE)

# First non-empty argument wins (used to merge nested cfg with defaults).
pick <- function(...) {
  for (d in list(...)) {
    if (is.null(d)) {
      next
    }
    if (length(d) == 1L && is.na(d)) {
      next
    }
    if (is.character(d) && !nzchar(d[1])) {
      next
    }
    return(d)
  }
  NULL
}

# Country / sex / ages / calendar years: yearsFit = fit_period (fixed window for metadata + weights).
hmd_cfg <- cfg$hmd
country <- pick(
  if (!is.null(hmd_cfg)) hmd_cfg$country else NULL,
  cfg$country,
  "SWE"
)
sex <- pick(
  if (!is.null(hmd_cfg)) hmd_cfg$sex else NULL,
  cfg$gender,
  "male"
)
ages_cfg <- cfg$ages
ages_min <- as.integer(pick(if (!is.null(ages_cfg)) ages_cfg$min else NULL, 50L))
ages_max <- as.integer(pick(if (!is.null(ages_cfg)) ages_cfg$max else NULL, 89L))
agesFit <- seq.int(ages_min, ages_max)
fp <- cfg$fit_period
fit_start <- as.integer(pick(if (!is.null(fp)) fp$start else NULL, 1960L))
fit_end <- as.integer(pick(if (!is.null(fp)) fp$end else NULL, 1990L))
yearsFit <- seq.int(fit_start, fit_end)

# Last calendar year we need realized rates for; rolling training ends at roll_b.
maxYear <- as.integer(pick(cfg$forecast_max_year, 2016L))
ro <- cfg$rolling_origins
roll_a <- as.integer(pick(if (!is.null(ro)) ro$start else NULL, 1990L))
roll_b <- as.integer(pick(if (!is.null(ro)) ro$end else NULL, 2015L))

# Horizons 1..H_MAX for weight estimation and for MSE reporting (cap forecast length).
H_MAX <- as.integer(pick(cfg$mse_horizon_max, 15L))
if (is.na(H_MAX) || H_MAX < 1L) {
  stop("mse_horizon_max must be a positive integer")
}

mcs_seed <- as.integer(pick(cfg$mcs_seed, 100L))
set.seed(mcs_seed)

# MCS bootstrap size, block length for tsboot, inclusion threshold alpha; holdout for single-validation methods.
mcs_B <- as.integer(pick(cfg$mcs_B, 10000L))
if (is.na(mcs_B) || mcs_B < 1L) {
  mcs_B <- 10000L
}
mcs_l <- as.integer(pick(cfg$mcs_block_l, 3L))
if (is.na(mcs_l) || mcs_l < 1L) {
  mcs_l <- 3L
}
mcs_alpha <- as.numeric(pick(cfg$mcs_alpha, 0.1))

# CoMoMo default-style tail holdout for bma(..., method = "sv") and mcs(..., method = "sv").
holdout_sv <- as.integer(round(length(yearsFit) / 3))
run_mode <- tolower(as.character(pick(cfg$run_mode, Sys.getenv("MORTALITY_RUN_MODE", unset = "evaluation"))))
allowed_run_modes <- c("evaluation", "forecast", "both")
if (!run_mode %in% allowed_run_modes) {
  stop("run_mode must be one of: ", paste(allowed_run_modes, collapse = ", "))
}
# If effective JSON sets forecast_target_year, ignore MORTALITY_FORECAST_HORIZON from .env/shell
# (otherwise e.g. .env MORTALITY_FORECAST_HORIZON=1 wins in pick() before the post-HMD block runs).
fty_probe <- suppressWarnings(as.integer(unlist(cfg[["forecast_target_year"]], use.names = FALSE)))
fty_probe <- fty_probe[!is.na(fty_probe)]
has_forecast_target_year <- length(fty_probe) >= 1L
env_fh <- Sys.getenv("MORTALITY_FORECAST_HORIZON", unset = "")
if (has_forecast_target_year && nzchar(env_fh)) {
  env_fh <- ""
}
forecast_horizon <- as.integer(pick(cfg$forecast_horizon, env_fh, H_MAX))
if (is.na(forecast_horizon) || forecast_horizon < 1L) {
  stop("forecast_horizon must be a positive integer")
}
forecast_horizon <- min(forecast_horizon, H_MAX)

model_ids <- unlist(cfg$models, use.names = FALSE)
if (length(model_ids) < 2L) {
  stop("config 'models' must contain at least 2 entries.")
}
allowed_models <- c("LC", "RH", "APC", "CBD", "M7", "PLAT")
unknown <- setdiff(model_ids, allowed_models)
if (length(unknown) > 0L) {
  stop(
    "Unknown model(s): ",
    paste(unknown, collapse = ", "),
    ". Allowed: ",
    paste(allowed_models, collapse = ", ")
  )
}

# ---- Download mortality and wrap as StMoMoData (deaths/exposures matrices + ages/years) ----
MorData <- hmd.mx(country = country, username = hmd_user, password = hmd_pass)
DataStMoMo <- StMoMoData(MorData, sex)

# If JSON sets forecast_target_year, derive horizon from data actually downloaded (latest HMD year).
fty_cfg <- cfg[["forecast_target_year"]]
if (!is.null(fty_cfg)) {
  fty <- suppressWarnings(as.integer(unlist(fty_cfg)))
  fty <- fty[!is.na(fty)]
  if (length(fty) < 1L) {
    stop("forecast_target_year must be a single integer calendar year when set.", call. = FALSE)
  }
  fty <- as.integer(fty[[1]])
  last_hmd_y <- max(DataStMoMo$years)
  fh_raw <- fty - last_hmd_y
  if (fh_raw < 1L) {
    stop(
      "forecast_target_year (", fty, ") must be after last HMD year in downloaded data (",
      last_hmd_y, ").",
      call. = FALSE
    )
  }
  forecast_horizon <- min(as.integer(fh_raw), H_MAX)
  if (fh_raw > H_MAX) {
    warning(
      "Horizon implied by forecast_target_year is ", fh_raw, "; capping forecast_horizon to H_MAX (",
      H_MAX, ")."
    )
  }
}

# All HMD calendar years from fit_start through latest year in data (subset of DataStMoMo$years).
years_fit_full <- DataStMoMo$years[DataStMoMo$years >= fit_start]
if (length(years_fit_full) < 2L) {
  stop("Need at least two HMD years with year >= fit_start for full-sample forecast.")
}

# ---- StMoMo GAPC objects: full registry; run subset given by config models ----
LC <- lc()
APC <- apc()
CBD <- cbd(link = "log")
M7 <- m7(link = "log")
RH <- rh(approxConst = TRUE)
f2 <- function(x, ages) mean(ages) - x
constPlat <- function(ax, bx, kt, b0x, gc, wxt, ages) {
  nYears <- dim(wxt)[2]
  x <- ages
  t <- 1:nYears
  c <- (1 - tail(ages, 1)):(nYears - ages[1])
  xbar <- mean(x)
  phiReg <- lm(gc ~ 1 + c + I(c^2), na.action = na.omit)
  phi <- coef(phiReg)
  gc <- gc - phi[1] - phi[2] * c - phi[3] * c^2
  kt[2, ] <- kt[2, ] + 2 * phi[3] * t
  kt[1, ] <- kt[1, ] + phi[2] * t + phi[3] * (t^2 - 2 * xbar * t)
  ax <- ax + phi[1] - phi[2] * x + phi[3] * x^2
  ci <- rowMeans(kt, na.rm = TRUE)
  ax <- ax + ci[1] + ci[2] * (xbar - x)
  kt[1, ] <- kt[1, ] - ci[1]
  kt[2, ] <- kt[2, ] - ci[2]
  list(ax = ax, bx = bx, kt = kt, b0x = b0x, gc = gc)
}
PLAT <- StMoMo(
  link = "log", staticAgeFun = TRUE, periodAgeFun = c("1", f2),
  cohortAgeFun = "1", constFun = constPlat
)

model_registry <- list(
  LC = LC, RH = RH, APC = APC, CBD = CBD, M7 = M7, PLAT = PLAT
)
modlist <- stats::setNames(
  lapply(model_ids, function(nm) model_registry[[nm]]),
  model_ids
)
modelNames <- model_ids

parse_stack_metalearners <- function(cfg) {
  env_raw <- Sys.getenv("MORTALITY_STACK_METALEARNERS", unset = "")
  raw <- cfg[["stack_metalearners"]]
  if (is.null(raw)) {
    if (nzchar(env_raw)) {
      raw <- trimws(strsplit(env_raw, ",", fixed = TRUE)[[1L]])
    } else {
      stop(
        "Set 'stack_metalearners' in the JSON config (array of strings) or ",
        "MORTALITY_STACK_METALEARNERS (comma-separated).\n",
        "Each entry must be one of: nnls, linear, lasso, ridge, elastic.\n",
        "Use an empty JSON array [] to run without stacked-regression combinations.",
        call. = FALSE
      )
    }
  } else {
    raw <- unlist(raw, use.names = FALSE)
    raw <- as.character(raw)
  }
  raw <- raw[!is.na(raw)]
  raw <- trimws(raw)
  raw <- raw[nzchar(raw)]
  tolower(raw)
}

validate_required_columns <- function(df, required, label) {
  missing_cols <- setdiff(required, names(df))
  if (length(missing_cols) > 0L) {
    stop(label, " is missing required columns: ", paste(missing_cols, collapse = ", "))
  }
}

validate_no_na_key <- function(df, keys, label) {
  bad <- stats::complete.cases(df[, keys, drop = FALSE])
  if (!all(bad)) {
    warning(label, ": dropping ", sum(!bad), " row(s) with NA in key columns.")
  }
  df[bad, , drop = FALSE]
}

# ---- Step 1 (fixed yearsFit): cross-validated metadata, then learn weights per method ----
# Used for rolling MSE / mortality evaluation only. Weights do NOT roll here; only refits in
# forecast_one_origin roll.
cat("Learning combination weights via CoMoMo (fixed window yearsFit, evaluation)...\n")
meta_eval <- CoMoMo::metadata(
  modlist,
  data = DataStMoMo,
  ages.fit = agesFit,
  years.fit = yearsFit,
  h = H_MAX
)

stack_ml_def <- list(
  nnls = list(co = "nnls", label = "stack_nnls"),
  linear = list(co = "Linear", label = "stack_linear"),
  lasso = list(co = "Lasso", label = "stack_lasso"),
  ridge = list(co = "Ridge", label = "stack_ridge"),
  elastic = list(co = "Elastic", label = "stack_elastic")
)
stack_keys <- parse_stack_metalearners(cfg)
allowed_stack <- names(stack_ml_def)
unknown <- setdiff(stack_keys, allowed_stack)
if (length(unknown) > 0L) {
  stop(
    "Unknown stack_metalearners: ",
    paste(unknown, collapse = ", "),
    ". Allowed: ",
    paste(allowed_stack, collapse = ", "),
    call. = FALSE
  )
}
if (length(stack_keys) != length(unique(stack_keys))) {
  stop("stack_metalearners must not contain duplicates.", call. = FALSE)
}

# nnls requires a finite design matrix; some model sets (e.g. M7+PLAT) or age windows
# can yield Inf/NaN in CV-based metadata -> nnls stops. Linear stack is more tolerant.
stack_weights_from_meta <- function(meta_obj, k) {
  d <- stack_ml_def[[k]]
  fit_stack <- function(co) {
    CoMoMo::stack(meta_obj, metalearner = co, normalize = TRUE, dynamic = TRUE)
  }
  tryCatch(
    fit_stack(d$co),
    error = function(e) {
      if (identical(d$co, "nnls")) {
        warning(
          "stack nnls failed: ", conditionMessage(e), ". Retrying with Linear for ", k, ".",
          call. = FALSE
        )
        fit_stack("Linear")
      } else {
        stop(e)
      }
    }
  )
}

stack_specs_eval <- lapply(stack_keys, function(k) {
  d <- stack_ml_def[[k]]
  list(w = stack_weights_from_meta(meta_eval, k), label = d$label)
})

# Each entry: CoMoMo weight object + short name used as factor level in output tables.
weight_specs_eval <- c(
  stack_specs_eval,
  list(
    # BMA: cv = weights from CV MSE; sv = weights from log-error on a single tail holdout.
    list(w = CoMoMo::bma(modlist, method = "cv", data = DataStMoMo, ages.fit = agesFit, years.fit = yearsFit, h = H_MAX), label = "bma_cv"),
    list(w = CoMoMo::bma(modlist, method = "sv", data = DataStMoMo, ages.fit = agesFit, years.fit = yearsFit, h = H_MAX, holdout = holdout_sv), label = "bma_sv"),
    # MCS: cv = per-horizon sets from CV losses; sv = one validation window; equal weights inside the set.
    list(w = CoMoMo::mcs(modlist, method = "cv", data = DataStMoMo, ages.fit = agesFit, years.fit = yearsFit, h = H_MAX, B = mcs_B, l = mcs_l, alpha = mcs_alpha), label = "mcs_cv"),
    list(w = CoMoMo::mcs(modlist, method = "sv", data = DataStMoMo, ages.fit = agesFit, years.fit = yearsFit, h = H_MAX, B = mcs_B, l = mcs_l, alpha = mcs_alpha, holdout = holdout_sv), label = "mcs_sv"),
    # Weights proportional to CV MSE (not the same as BMA cv exp(-0.5 * mse) scaling).
    list(w = CoMoMo::frequentist(modlist, data = DataStMoMo, ages.fit = agesFit, years.fit = yearsFit, h = H_MAX), label = "frequentist")
  )
)

# Same methods as fitModels_full / future forecast: metadata + weights on years_fit_full.
holdout_sv_full <- as.integer(round(length(years_fit_full) / 3))
meta_full <- NULL
weight_specs_full <- NULL
if (run_mode %in% c("forecast", "both")) {
  cat("Learning combination weights for future forecast (years.fit = years_fit_full)...\n")
  meta_full <- CoMoMo::metadata(
    modlist,
    data = DataStMoMo,
    ages.fit = agesFit,
    years.fit = years_fit_full,
    h = H_MAX
  )
  stack_specs_full <- lapply(stack_keys, function(k) {
    d <- stack_ml_def[[k]]
    list(w = stack_weights_from_meta(meta_full, k), label = d$label)
  })
  weight_specs_full <- c(
    stack_specs_full,
    list(
      list(w = CoMoMo::bma(modlist, method = "cv", data = DataStMoMo, ages.fit = agesFit, years.fit = years_fit_full, h = H_MAX), label = "bma_cv"),
      list(w = CoMoMo::bma(modlist, method = "sv", data = DataStMoMo, ages.fit = agesFit, years.fit = years_fit_full, h = H_MAX, holdout = holdout_sv_full), label = "bma_sv"),
      list(w = CoMoMo::mcs(modlist, method = "cv", data = DataStMoMo, ages.fit = agesFit, years.fit = years_fit_full, h = H_MAX, B = mcs_B, l = mcs_l, alpha = mcs_alpha), label = "mcs_cv"),
      list(w = CoMoMo::mcs(modlist, method = "sv", data = DataStMoMo, ages.fit = agesFit, years.fit = years_fit_full, h = H_MAX, B = mcs_B, l = mcs_l, alpha = mcs_alpha, holdout = holdout_sv_full), label = "mcs_sv"),
      list(w = CoMoMo::frequentist(modlist, data = DataStMoMo, ages.fit = agesFit, years.fit = years_fit_full, h = H_MAX), label = "frequentist")
    )
  )
}

combination_methods <- c("average", vapply(weight_specs_eval, function(s) s$label, character(1)))

# Export-friendly tables so plotting scripts do not need to recompute CoMoMo weights/CV.
cvmse_by_horizon_out <- meta_eval$cvmse %>%
  dplyr::transmute(
    h = as.integer(h),
    model = as.character(model),
    cvmse = as.numeric(model.cvmse)
  ) %>%
  dplyr::arrange(h, cvmse)

weights_all_methods_out <- dplyr::bind_rows(
  dplyr::tibble(
    h = rep(seq_len(H_MAX), each = length(modelNames)),
    model = rep(modelNames, times = H_MAX),
    model.weights = 1 / length(modelNames),
    method = "average"
  ),
  dplyr::bind_rows(lapply(weight_specs_eval, function(spec) {
    spec$w$weights %>%
      dplyr::transmute(
        h = as.integer(h),
        model = as.character(model),
        model.weights = as.numeric(model.weights),
        method = as.character(spec$label)
      )
  }))
) %>%
  dplyr::arrange(method, h, dplyr::desc(model.weights))

# Figure 6 guardrails: validate key schema, remove bad rows, and complete missing cells.
validate_required_columns(
  weights_all_methods_out,
  c("h", "model", "model.weights", "method"),
  "weights_all_methods_out"
)
weights_all_methods_out <- validate_no_na_key(
  weights_all_methods_out,
  c("h", "model", "method"),
  "weights_all_methods_out"
)
weights_all_methods_out <- weights_all_methods_out %>%
  dplyr::mutate(
    h = as.integer(h),
    model = as.character(model),
    method = as.character(method),
    model.weights = as.numeric(model.weights)
  ) %>%
  tidyr::complete(
    method = combination_methods,
    h = seq_len(H_MAX),
    model = modelNames,
    fill = list(model.weights = 0)
  ) %>%
  dplyr::arrange(method, h, dplyr::desc(model.weights))

if (any(!is.finite(weights_all_methods_out$model.weights))) {
  warning("weights_all_methods_out has non-finite weights; coercing to 0 for plotting.")
  weights_all_methods_out$model.weights[!is.finite(weights_all_methods_out$model.weights)] <- 0
}

out_root <- Sys.getenv("MORTALITY_OUTPUT_DIR", unset = "output/comomo_new_fixed_forecast")
run_ts <- format(Sys.time(), "%Y%m%d_%H%M%S")
out_dir <- file.path(out_root, paste0("run_", run_ts))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_ts_csv <- function(df, stem) {
  write.csv(
    df,
    file.path(out_dir, paste0(stem, "_", run_ts, ".csv")),
    row.names = FALSE
  )
}

save_ts_plot <- function(plot_obj, stem, w = 11, h = 6.5) {
  png_path <- file.path(out_dir, paste0(stem, "_", run_ts, ".png"))
  pdf_path <- file.path(out_dir, paste0(stem, "_", run_ts, ".pdf"))
  ggplot2::ggsave(png_path, plot_obj, width = w, height = h, dpi = 300)
  ggplot2::ggsave(pdf_path, plot_obj, width = w, height = h)
}

# CoMoMo::forecast.CoMoMo can mis-truncate weights when max(weights$h) > h; align rows to h_use.
prepare_weight_for_forecast_horizon <- function(w, h_use) {
  if (!inherits(w, "weight") || is.null(w$weights) || nrow(w$weights) < 1L) {
    return(w)
  }
  wm <- as.data.frame(w$weights, stringsAsFactors = FALSE)
  need <- c("h", "model", "model.weights")
  if (!all(need %in% names(wm))) {
    return(w)
  }
  h_use <- as.integer(h_use)
  if (!is.finite(h_use) || h_use < 1L) {
    return(w)
  }
  wm$h <- as.integer(wm$h)
  max_h <- max(wm$h, na.rm = TRUE)
  if (!is.finite(max_h) || max_h < 1L) {
    return(w)
  }
  last <- wm[wm$h == max_h, , drop = FALSE]
  if (max_h < h_use) {
    if (nrow(last) < 1L) {
      return(w)
    }
    pad <- do.call(
      rbind,
      lapply(seq.int(max_h + 1L, h_use), function(hh) {
        data.frame(
          h = hh,
          model = last$model,
          model.weights = as.numeric(last$model.weights),
          stringsAsFactors = FALSE
        )
      })
    )
    wm <- rbind(wm, pad)
  } else if (max_h > h_use) {
    wm <- wm[wm$h <= h_use, , drop = FALSE]
  }
  w2 <- w
  w2$weights <- wm
  w2
}

# w_specs: list of list(w=..., label=...) from weight_specs_eval (rolling) or weight_specs_full (future).
forecast_from_fits <- function(fitModels, h_use, w_specs) {
  if (missing(w_specs) || is.null(w_specs)) {
    stop("forecast_from_fits: w_specs is required (pass weight_specs_eval or weight_specs_full).", call. = FALSE)
  }
  # weight = NULL gives class CoMoMo.simple: base rates + log-average combined rate.
  fc_sma <- forecast::forecast(CoMoMo::CoMoMo(fitModels, weight = NULL), h = h_use)
  # CoMoMo uses column name "ages"; downstream joins use "age".
  base_df <- fc_sma$model.rates %>%
    dplyr::rename(age = ages)

  comb_avg <- fc_sma$comb.rates %>%
    dplyr::rename(age = ages) %>%
    dplyr::mutate(model = as.character(model))

  ensemble_parts <- list(comb_avg)
  for (spec in w_specs) {
    # One combined forecast per weight object (log-linear pool with method-specific weights).
    w_safe <- prepare_weight_for_forecast_horizon(spec$w, h_use)
    fc_w <- forecast::forecast(CoMoMo::CoMoMo(fitModels, weight = w_safe), h = h_use)
    ensemble_parts[[length(ensemble_parts) + 1L]] <- fc_w$comb.rates %>%
      dplyr::rename(age = ages) %>%
      dplyr::mutate(model = spec$label)
  }

  dplyr::bind_rows(base_df, dplyr::bind_rows(ensemble_parts))
}

# ---- Step 2 (rolling): expand training to fit_start:yearFitEval (yearsFit[1] == fit_start); forecast h_use ahead ----
# Same expanding window as rolling_window_check_mse_comomo_api_fixed.R.
# Cap h_use at H_MAX so horizon-specific weights always exist (avoids fragile paths in forecast.CoMoMo).
forecast_one_origin <- function(yearFitEval) {
  h_use <- min(as.integer(maxYear - yearFitEval), H_MAX)
  if (!is.finite(h_use) || h_use < 1L) {
    return(NULL)
  }

  # Named list of StMoMo fits on the expanding window (required by CoMoMo::CoMoMo).
  fitModels <- stats::setNames(
    CoMoMo::fitCoMoMo(
      modlist,
      data = DataStMoMo,
      ages.fit = agesFit,
      years.fit = yearsFit[1]:yearFitEval
    ),
    modelNames
  )
  forecast_from_fits(fitModels, h_use = h_use, w_specs = weight_specs_eval)
}

# Write always-available metadata tables (needed by plotting and forecast mode).
cvmse_table <- cvmse_by_horizon_out %>%
  dplyr::mutate(
    h = as.integer(h),
    model = as.character(model),
    cvmse = round(as.numeric(cvmse), 6)
  ) %>%
  dplyr::arrange(h, cvmse)
weights_table <- weights_all_methods_out %>%
  dplyr::transmute(
    h = as.integer(h),
    method = as.character(method),
    model = as.character(model),
    weight = round(as.numeric(model.weights), 8)
  ) %>%
  dplyr::arrange(method, h, dplyr::desc(weight))
write_ts_csv(cvmse_by_horizon_out, "cvmse_by_horizon_base_models")
write_ts_csv(weights_all_methods_out, "weights_all_methods_by_horizon")
write_ts_csv(cvmse_table, "cvmse_table")
write_ts_csv(weights_table, "weights_table")

if (run_mode %in% c("evaluation", "both")) {
  # Stack all rolling origins (long table: age, year, h, model, rate).
  cat("Rolling forecasts (CoMoMo::fitCoMoMo + forecast)...\n")
  forRatesHor <- dplyr::bind_rows(
    lapply(seq.int(roll_a, roll_b), forecast_one_origin)
  )

  # Observed death rates Dxt/Ext in long form for joining to forecast years.
  obsRatesDF <- tidyr::pivot_longer(
    dplyr::bind_cols(age = DataStMoMo$ages, as.data.frame(DataStMoMo$Dxt / DataStMoMo$Ext)),
    cols = 2:(length(DataStMoMo$years) + 1),
    names_to = "year",
    values_to = "obsrate"
  ) %>%
    dplyr::mutate(year = as.numeric(year))

  allForecast <- dplyr::left_join(forRatesHor, obsRatesDF, by = c("age", "year"))

  # Mean squared log error by horizon and model (base + each ensemble label).
  mseDF <- allForecast %>%
    dplyr::filter(h <= H_MAX) %>%
    dplyr::mutate(err2 = (log(rate) - log(obsrate))^2) %>%
    dplyr::group_by(h, model) %>%
    dplyr::summarise(mse = mean(err2, na.rm = TRUE), .groups = "drop")

  # Pointwise forecasts with the global MSE(h, model) repeated on each row for convenience.
  forecast_rates_mse_out <- allForecast %>%
    dplyr::filter(h <= H_MAX) %>%
    dplyr::mutate(err2 = (log(rate) - log(obsrate))^2) %>%
    dplyr::left_join(mseDF, by = c("h", "model")) %>%
    dplyr::select(age, year, h, model, rate, obsrate, err2, mse) %>%
    dplyr::arrange(age, year, h, model)

  mseDF_out <- mseDF %>%
    dplyr::mutate(mse = round(mse, 6)) %>%
    dplyr::arrange(h, mse)

  avgMSE <- mseDF %>%
    dplyr::group_by(model) %>%
    dplyr::summarise(avg_mse = mean(mse, na.rm = TRUE), .groups = "drop") %>%
    dplyr::mutate(avg_mse = round(avg_mse, 6)) %>%
    dplyr::arrange(avg_mse) %>%
    dplyr::mutate(Ranking = dplyr::row_number())

  mse_table <- mseDF %>%
    dplyr::transmute(
      h = as.integer(h),
      model = as.character(model),
      mse = round(as.numeric(mse), 6)
    ) %>%
    dplyr::arrange(h, mse)

  mse_summary_table <- avgMSE %>%
    dplyr::transmute(
      model = as.character(model),
      avg_mse = round(as.numeric(avg_mse), 6),
      ranking = as.integer(Ranking)
    ) %>%
    dplyr::arrange(ranking, avg_mse)

  cat("\n=== MSE by horizon and model (using CoMoMo) ===\n")
  print(mseDF_out, n = nrow(mseDF_out))
  cat("\n=== Average MSE across horizons ===\n")
  print(avgMSE, n = nrow(avgMSE))

  write_ts_csv(mseDF_out, "mse_by_horizon_output_comomo_api")
  write_ts_csv(avgMSE, "mse_average_ranking_output_comomo_api")
  write_ts_csv(forecast_rates_mse_out, "forecast_rates_with_mse_output_comomo_api")
  write_ts_csv(mse_table, "mse_table")
  write_ts_csv(mse_summary_table, "mse_summary_table")
}

if (run_mode %in% c("forecast", "both")) {
  cat(
    "\nGenerating future forecasts from full HMD fit (years ",
    min(years_fit_full), ":", max(years_fit_full), ")...\n",
    sep = ""
  )
  fitModels_full <- stats::setNames(
    CoMoMo::fitCoMoMo(
      modlist,
      data = DataStMoMo,
      ages.fit = agesFit,
      years.fit = years_fit_full
    ),
    modelNames
  )
  if (is.null(weight_specs_full)) {
    stop("weight_specs_full is NULL; forecast mode requires learned weights on years_fit_full.", call. = FALSE)
  }
  forecast_future_out <- forecast_from_fits(
    fitModels_full,
    h_use = forecast_horizon,
    w_specs = weight_specs_full
  ) %>%
    dplyr::transmute(
      age = as.integer(age),
      year = as.integer(year),
      h = as.integer(h),
      model = as.character(model),
      rate = as.numeric(rate)
    ) %>%
    dplyr::arrange(year, h, age, model)
  write_ts_csv(forecast_future_out, "forecast_future_rates")

  forecast_plot_df <- forecast_future_out %>%
    dplyr::group_by(h, model) %>%
    dplyr::summarise(
      year = dplyr::first(year),
      rate = mean(rate, na.rm = TRUE),
      .groups = "drop"
    )

  p_forecast <- ggplot2::ggplot(
    forecast_plot_df,
    ggplot2::aes(x = h, y = rate, color = model, group = model)
  ) +
    ggplot2::geom_line(linewidth = 0.9) +
    ggplot2::geom_point(size = 1.5) +
    ggplot2::theme_bw(base_size = 10) +
    ggplot2::labs(
      title = paste0("Future mortality forecast by horizon (", country, ", ", sex, ")"),
      subtitle = paste0(
        "Full HMD fit ", min(years_fit_full), "-", max(years_fit_full),
        "; mean across ages; h=1 is year ", max(years_fit_full) + 1L
      ),
      x = "Forecast horizon (h)",
      y = "Forecast rate",
      color = "Model"
    ) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(hjust = 0.5),
      plot.subtitle = ggplot2::element_text(hjust = 0.5),
      legend.position = "bottom"
    )
  save_ts_plot(p_forecast, "forecast_future_rates_by_h")
}

cat("\nWrote CSVs under ", normalizePath(out_dir, mustWork = FALSE), "\n", sep = "")
