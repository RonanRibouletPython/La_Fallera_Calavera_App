# Base image
FROM mcr.microsoft.com/devcontainers/python:3.13-bookworm

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \        
    tree \                
    lsof \  
    # Install Tesseract for POC
    tesseract-ocr \
    tesseract-ocr-cat \
    tesseract-ocr-eng \
    tesseract-ocr-spa \              
    && rm -rf /var/lib/apt/lists/*



# Package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

USER vscode
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/vscode/.local/bin:${PATH}"

WORKDIR /workspace
CMD ["sleep", "infinity"]