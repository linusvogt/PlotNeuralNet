# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

PlotNeuralNet generates TikZ/LaTeX diagrams of neural network architectures. A diagram is defined either via the Python API (which generates a `.tex` file) or by writing `.tex` files directly. The `.tex` file is compiled to PDF with `pdflatex`.

## Building

To compile a diagram, run from the directory containing the script:

```bash
cd pyexamples/
bash ../tikzmake.sh test_simple   # runs test_simple.py → test_simple.tex → test_simple.pdf
```

`tikzmake.sh` runs the Python script, compiles with `pdflatex`, cleans auxiliary files, and opens the PDF.

To compile a `.tex` file directly (skipping Python):
```bash
pdflatex mydiagram.tex
```

**Dependencies:** Python (no third-party packages), and LaTeX:
```bash
sudo apt-get install texlive-latex-extra   # Ubuntu 16.04
# Ubuntu 18.04+: texlive-latex-base texlive-fonts-recommended texlive-fonts-extra texlive-latex-extra
```

There are no automated tests or linting configured.

---

## Writing `.tex` Files Directly (Recommended for Full Control)

Writing `.tex` directly is more powerful than the Python API — it exposes all TikZ parameters including 3D coordinate routing, custom colors, opacity, image backgrounds, and arbitrary math in node labels.

### File Template

```latex
\documentclass[border=8pt, multi, tikz]{standalone}
\usepackage{import}
\subimport{../../layers/}{init}   % adjust relative path to layers/
\usetikzlibrary{positioning}
\usetikzlibrary{3d}

% Color definitions (mix freely using rgb: syntax)
\def\ConvColor{rgb:yellow,5;red,2.5;white,5}
\def\ConvReluColor{rgb:yellow,5;red,5;white,5}
\def\PoolColor{rgb:red,1;black,0.3}
\def\UnpoolColor{rgb:blue,2;green,1;black,0.3}
\def\FcColor{rgb:blue,5;red,2.5;white,5}
\def\SumColor{rgb:blue,5;green,15}
\def\edgecolor{rgb:yellow,5;red,2.5;white,5}

\newcommand{\midarrow}{\tikz \draw[-Stealth,line width=0.8mm,draw=\edgecolor] (-0.3,0) -- ++(0.3,0);}
\newcommand{\copymidarrow}{\tikz \draw[-Stealth,line width=0.8mm,draw={rgb:blue,4;red,1;green,1;black,3}] (-0.3,0) -- ++(0.3,0);}

\begin{document}
\begin{tikzpicture}
\tikzstyle{connection}=[ultra thick,every node/.style={sloped,allow upside down},draw=\edgecolor,opacity=0.7]
\tikzstyle{copyconnection}=[ultra thick,every node/.style={sloped,allow upside down},draw={rgb:blue,4;red,1;green,1;black,3},opacity=0.7]

% ... layers and connections here ...

\end{tikzpicture}
\end{document}
```

### Layer Types

#### Box — simple rectangular layer (Conv, Pool, FC, etc.)
```latex
\pic[shift={(x,y,z)}] at (anchor)
    {Box={
        name=layername,
        caption=Label,
        xlabel={{64}},       % filter count label on front face; use {{64, dummy}} for padding
        ylabel=224,          % height label
        zlabel=224,          % depth label
        fill=\ConvColor,
        opacity=0.5,
        height=40,
        width=2,             % visual thickness proportional to filter count
        depth=40
    }};
```

#### RightBandedBox — dual-color box (Conv+ReLU visualization)
```latex
\pic[shift={(x,y,z)}] at (anchor)
    {RightBandedBox={
        name=layername,
        caption=Conv+ReLU,
        xlabel={{"64","64"}},     % must be array of 2
        fill=\ConvColor,
        bandfill=\ConvReluColor,  % right-band color
        opacity=0.4,
        bandopacity=0.6,
        height=40,
        width={2,2},              % must be array of 2
        depth=40
    }};
```

#### Ball — sphere node (Sum, Concat, etc.)
```latex
\pic[shift={(x,y,z)}] at (anchor)
    {Ball={
        name=layername,
        caption=Sum,
        fill=\SumColor,
        radius=2.5,
        opacity=0.6,
        logo=$+$        % any math expression; use $||$ for concatenation
    }};
```

### Available Node Coordinates

All Box/RightBandedBox nodes expose:
- Faces: `name-west`, `name-east`, `name-north`, `name-south`
- 3D corners: `name-nearnortheast`, `name-farnortheast`, `name-nearsoutheast`, `name-farsoutheast`, etc. (8 total)

Ball nodes expose: `name-west`, `name-east`, `name-north`, `name-south`, `name-anchor`

### Connections

```latex
% Standard forward connection
\draw [connection] (layer1-east) -- node {\midarrow} (layer2-west);

% Skip/copy connection routed above layers
\path (cr4-southeast) -- (cr4-northeast) coordinate[pos=1.25] (cr4-top);
\path (cat4-south) -- (cat4-north) coordinate[pos=1.25] (cat4-top);
\draw [copyconnection] (cr4-northeast)
    -- node {\copymidarrow} (cr4-top)
    -- node {\copymidarrow} (cat4-top)
    -- node {\copymidarrow} (cat4-north);

% Orthogonal routing with intermediate coordinate
\path (p4-east) -- (cr5-west) coordinate[pos=0.25] (between4_5);
\draw [connection] (between4_5)
    -- node {\midarrow} (score16-west-|between4_5)
    -- node {\midarrow} (score16-west);
```

`pos=1.25` places a coordinate 25% beyond the end of a path segment — useful for routing above/below layers.

### Positioning

```latex
% First layer at origin
\pic[shift={(0,0,0)}] at (0,0,0) {Box={name=conv1, ...}};

% Subsequent layer 1 unit to the right
\pic[shift={(1,0,0)}] at (conv1-east) {Box={name=pool1, ...}};

% Layer with vertical offset (e.g., decoder path)
\pic[shift={(1.2,-10,0)}] at (prev-east) {Box={name=dec1, ...}};
```

### Advanced Features

```latex
% Include a background image on the YZ plane
\node[canvas is zy plane at x=0] (img) at (-3,0,0)
    {\includegraphics[width=8cm,height=8cm]{input.jpg}};

% Transparent grouping/annotation box
\pic[shift={(0,0,0)}] at (anchor)
    {Box={name=group, caption=, fill=, opacity=0.2, height=42, width={8}, depth=42}};
```

---

## Python API (pycore/)

The Python API (`pycore/tikzeng.py`) is a convenience wrapper that generates the LaTeX above. Each function returns a raw string; `to_generate(arch, 'output.tex')` concatenates and writes the file.

```python
import sys; sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import *   # composite helpers: block_2ConvPool, block_Unconv, block_Res

arch = [
    to_head('..'),     # path to layers/ directory
    to_cor(),
    to_begin(),
    to_Conv("conv1", s_filer=32, n_filer=64, offset="(0,0,0)", to="(0,0,0)", height=40, depth=40, width=2),
    to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)"),
    to_connection("conv1", "pool1"),
    to_end()
]
to_generate(arch, 'output.tex')
```

Key `to_Conv` parameters: `name`, `s_filer` (spatial size label), `n_filer` (filter count label), `offset` (3D shift), `to` (anchor), `height`, `depth`, `width`, `caption`.

The Python API does **not** expose: per-layer opacity, `bandopacity`, all 3D corner coordinates, image inclusion, custom Ball logos, or advanced path routing. Use raw LaTeX for those.


**IMPORTANT** do not use the Python API, instead directly generate LaTeX code. This gives more granular control over all elements in the figure and allows for more detailed troubleshooting.