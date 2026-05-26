rm -rf /tmp/wm_viz_out && \
  uv run python -m datagen.generate_raw \
    --num-episodes 2 --output-dir /tmp/wm_viz_smoke \
    --episodes-per-shard 5 --seed 7 2>&1 \
  | tail -5 && echo "---" && \
  uv run python -m viz.cli --input /tmp/wm_viz_smoke \
    --output /tmp/wm_viz_out --downsample 4 2>&1 \
  | tail -10 && echo "---" && ls -lh /tmp/wm_viz_out/