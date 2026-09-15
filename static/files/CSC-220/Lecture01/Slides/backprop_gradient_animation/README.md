# One-weight backpropagation animation

The animation follows one connection through a complete learning step:

1. Forward pass: `x = 0.80`, `w = 0.50`, and `â = sigmoid(wx) ≈ 0.60`.
2. Squared-error loss against `y = 1.00`: `L = 0.5(â-y)^2 ≈ 0.081`.
3. Chain rule: `dL/dw = (â-y) sigmoid'(z) x ≈ -0.077`.
4. Gradient descent with `eta = 1`: `w_new = 0.50 - 1(-0.077) ≈ 0.577`.
5. The new prediction rises to about `0.61`, and the loss falls to about `0.075`.

## Use in an existing Beamer deck

Add `\usepackage{animate}` to the preamble, copy the frame directory beside your
main TeX file, and insert the contents of `backprop_gradient_animation.tex`.

## Compile the standalone demo

```bash
pdflatex backprop_gradient_demo.tex
```

The controls work in PDF viewers that support PDF JavaScript animation, such as
Adobe Acrobat Reader. The included GIF provides a quick preview in browsers.
