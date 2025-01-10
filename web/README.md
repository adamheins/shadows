# Shadows in the Browser

This is a (limited) Javascript implementation of the shadows library.

## Install

Node is required to install dependencies:
```
npm install
```
Then compile the Javascript:
```
npx rollup --config
```

## Run locally

The onnx models need to be served locally. In one terminal, do:
```
cd onnx
npx http-server -p 8000 --cors
```

Then open `index.html` in your browser of choice and enjoy!
