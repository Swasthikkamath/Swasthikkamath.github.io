---
layout: page
title: Bibliography
permalink: /bib/
years: [1956, 1950, 1935, 1905]
nav: true
---

<div class="publications">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f bibliography -q @*[year={{y}}]* %}
{% endfor %}

</div>

