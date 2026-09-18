ARG RUNTIME_IMAGE=vllm-diffusiongemma-structured:pr57250-d2c2b54
FROM ${RUNTIME_IMAGE}
LABEL org.opencontainers.image.source="https://github.com/pst2154/Typesafe_Testing" \
      org.opencontainers.image.title="DiffusionGemma structured decision service" \
      org.opencontainers.image.description="Patched vLLM diffusion runtime with typed decisions and Decision Lab explorer" \
      org.opencontainers.image.version="pr57250-d2c2b54-fix1"
COPY structured_server.py /opt/vllm/examples/features/diffusion_reads/structured_server.py
COPY gateway.py explorer.html /opt/decision-lab/
COPY launch.sh /usr/local/bin/launch-decision-lab
COPY NOTICE /opt/decision-lab/NOTICE
RUN chmod 0755 /usr/local/bin/launch-decision-lab
ENV MOE_BACKEND=marlin GATEWAY_PORT=8012
EXPOSE 8012
ENTRYPOINT ["/usr/local/bin/launch-decision-lab"]
