# Live Reliable LLM mortality app: Python Streamlit + R CoMoMo/StMoMo
FROM rocker/r-ver:4.3.3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIPELINE_MODE=full \
    R_PIPELINE_TIMEOUT_SEC=28800

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    gfortran \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# R packages used by the CoMoMo rolling-window script
RUN R -e "install.packages(c('jsonlite','demography','dplyr','tidyr','forecast','ggplot2','pkgload','StMoMo'), repos='https://cloud.r-project.org')"

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py
COPY src /app/src
COPY config /app/config
COPY prompts /app/prompts
COPY examples /app/examples
COPY sample_data /app/sample_data
COPY r /app/r
COPY .streamlit /app/.streamlit

ENV R_PIPELINE_SCRIPT=/app/r/rolling_window_check_mse_comomo_api_fixed_forecast_final.R

EXPOSE 8501

HEALTHCHECK CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
