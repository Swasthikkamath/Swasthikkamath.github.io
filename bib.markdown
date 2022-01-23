---
layout: page
title: Bibliography
permalink: /bib/
years: [2014, 2015, 2016, 2017, 2018, 2019, 2020, 1998, 1975]
nav: true
---

<div class="publications">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f bibliography -q @*[year={{y}}]* %}
{% endfor %}

</div>

