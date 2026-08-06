---
layout: page
title: projects
permalink: /projects/
description: Selected projects and experiments.
nav: true
nav_order: 3
---

{% assign featured_projects = site.projects | where: 'featured', true | sort: 'importance' %}

<div class="projects">
  {% for project in featured_projects %}
    <article class="mb-4">
      <h3 class="mb-1">
        <a href="{% if project.redirect %}{{ project.redirect }}{% else %}{{ project.url | relative_url }}{% endif %}">
          {{ project.title }}
        </a>
        {% if project.year %}<small>({{ project.year }})</small>{% endif %}
      </h3>
      <p>
        {{ project.description }}
        {% if project.code %}[<a href="{{ project.code }}">code</a>]{% endif %}
      </p>
    </article>
  {% endfor %}
</div>
