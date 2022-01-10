---
layout: page
title: README
permalink: /readme/
---

Each of these pages, "about us" and "authors" are located in the base directory as markdown files.

All meta information is stored in `_config.yml`.

Using jekyll-spaceship for all media processing - gifs, videos(links), LaTeX (they are using [MathJax](www.mathjax.org)).

Using `_plugins/video_tag.rb` for local video embedding.

Github pages does not allow processing of external plugins, so static HTML needs to be generated locally and hosted on github.

> We might need to think about what to do for this. One option is to run a script to generate the html and push to the repo, or otherwise we have each person build locally and push.

Using [Commento](https://commento.io/) for comment section. I chose this because it allows complete anonymous posting of comments. We may or may not want this (to be discussed). Based on this we can choose to use some other commenting service, or to host it ourselves with our own DB and script. We would also need to discuss the pricing component of this.

All styling is in the `_sass` directory. Page layouts are defined in the `_layouts` directory. HTML macros are definded in the `_includes` directory.

Draft posts can be written in the `_drafts` directory, and will not be converted to HTML when the sie is built. They can then be copied into the `_posts` directory to be published.

