FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

ENV CC=gcc CXX=g++
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy minimal source tree and runtime scripts
COPY src/oss25 /app/src/oss25
COPY Process_SkillEval.sh /usr/local/bin/Process_SkillEval.sh
RUN chmod +x /usr/local/bin/Process_SkillEval.sh

# Include model weights for submission
COPY weight /app/weight

# Create placeholder for artifacts (mount or bake separately)
RUN mkdir -p /app/artifacts

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Entrypoint expects: /usr/local/bin/Process_SkillEval.sh <task>
CMD ["/usr/local/bin/Process_SkillEval.sh", "GRS"]
