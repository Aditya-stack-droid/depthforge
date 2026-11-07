.PHONY: run docker eval train-synth test
run:            ## local app on :7860
	python app.py
docker:         ## one-command run
	docker compose up --build
eval:           ## regenerate results/eval_P4_T01.md on NYU test (needs the .mat)
	python scripts/evaluate.py --data nyu --mat nyu_depth_v2_labeled.mat --n 100
train-synth:
	python scripts/train.py --data synth --minutes 15
test:
	python -m pytest -q tests
