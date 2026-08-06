---
layout: page
permalink: /blog/
title: blog
description: Notes and experiments.
nav: true
nav_order: 1
---

{% assign personal_posts = site.posts | where: 'personal', true %}

<div class="post">
  {% if personal_posts.size > 0 %}
    <div class="post-list">
      {% for post in personal_posts %}
        <div class="row mb-2">
          <div class="col-sm-3 text-muted">{{ post.date | date: '%b %d, %Y' }}</div>
          <div class="col-sm-9">
            <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
          </div>
        </div>
      {% endfor %}
    </div>
  {% else %}
    <p>No posts yet.</p>
  {% endif %}
</div>
