---
layout: page
title: Authors
permalink: /authors/
---

<style>

.author-block {
  height: 150px;
  padding-top: 30px;
  padding-bottom: 30px;
  padding-right: 15px;
  background-color: lightblue;
  width: 80%;
  margin-left: auto;
  margin-right: auto;
  display: flex;
  justify-content: center;
  align-items: center;
}

.left-section {
  width: 40%;
  height: 100%
}

.profile-pic {
  border-radius: 50%;
  width: 100px;
  height: 100px;
  margin-right: auto;
  margin-left: auto;
  overflow: hidden;
}

.view-articles {
  background-color: #f08080;
  width: 80%;
  border-radius: 10px;
  margin-top: 15px;
  height: 30%;
  margin-right: auto;
  margin-left: auto;
  display: flex;
  align-items: center;
  justify-content: center;
}

.view-articles:hover {
  background-color: #ef6f6f;
}

.text {
  width: 60%;
  
}

.name {
  font-weight: bold;
}

</style>

{% for author in site.authors %}
{% assign filtered_posts = site.posts | where_exp: 'post', "post.authors contains author.short_name" %}

<div class="author-block">

  <div class="left-section">

    <div class="profile-pic">
      <img src="/assets/authors/{{ author.profile_pic }}">
    </div>

    <div onclick="window.location.href = '{{ author.url | absolute_url }}';" class="view-articles">
      -> {{ filtered_posts.size }}
      {% if filtered_posts.size == 1 %}
        Article
      {% else %}
        Articles
      {%- endif -%}
     </div>

  </div>

  <div class="text">

    <div class="name">
      <a href="{{ author.url }}">{{ author.name }}</a>
    </div>

    <div class="bio">
      {{ author.excerpt }}
    </div>

  </div>

</div>

{% endfor %}
