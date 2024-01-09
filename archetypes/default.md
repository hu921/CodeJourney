---
title: "{{ replace .Name "-" " " | title }}"
date: {{ .Date }}
description : "Description goes here..."
tags: [""]
image : ""
draft: true
---
<!--
+++
title = '{{ replace .File.ContentBaseName "-" " " | title }}'
date = {{ .Date }}
draft = true
+++
-->