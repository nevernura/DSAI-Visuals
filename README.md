# DS & AI Visual Lab

A self-study Streamlit app for foundational data science and AI. Each page is a
small lesson built on one recipe: an interactive visualisation, the equation one
click away, a few guided tasks, and one "break it" moment where the method
visibly fails.

This build ships the **shared template** plus the **gradient descent** module as
the reference implementation. Every other module is added by copying that page.

## Link

https://dsai-visuals.streamlit.app

## Run locally

```bash
pip install -r requirements.txt
streamlit run Home.py
```

Then open the URL it prints (default http://localhost:8501).

## Deploy free on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Create an app, pick the repo, and set the main file to `Home.py`.
4. Deploy. Pushing new commits redeploys automatically.

The free tier sleeps when idle and has modest CPU/RAM, so the app is built to
stay light: dataset generators are cached, point counts are small, and the
descent animation runs in the browser (Plotly frames) rather than as a
server-side loop.

## Project layout

```
Home.py              entry point; sectioned sidebar navigation
views/               one file per lesson (home, gradient_descent, _template)
lib/
  components.py      the per-module template: intro, math, tasks, break-it
  datasets.py        cached toy data generators (blobs, noisy line, rings)
  descent.py         gradient descent maths (the part you swap per module)
  plotting.py        Plotly figure builders
.streamlit/          theme + server config
```

## Add a new module

1. Copy `views/_template.py` to `views/<module>.py`.
2. Put the maths in `lib/<module>.py`, mirroring `lib/descent.py`.
3. Add figure building to `lib/plotting.py` if needed.
4. Fill in the template sections (controls, visualisation, maths, tasks, break-it).
5. Register it in `Home.py` under the right sidebar section.

Keeping every page to the same recipe is what makes this a kit rather than a
pile of one-off demos.

## Roadmap

- **Phase 1 (core path):** vectors & dot product, matrix multiplication,
  separability, gradient descent, linear regression, logistic regression.
- **Phase 2:** eigenvectors, PCA, SVM, decision trees, k-means.
- **Phase 3:** random forest, perceptron, MLP, convolution.
