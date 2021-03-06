---
title: Getting started guide
subtitle:
layout: page
toc: true
show_sidebar: false
hero_height: is-small
---

This guide is intended to help get Dev Days participants up-and-running with both:

1. the infrastructure for the 2021 Dev Days event, as well as
1. development environments for the nilearn and nibabel code bases.

## Sprint infrastructure

To create more dynamic time together (albeit remotely), we have created a [Nilearn-Nibabel-Dev-Days Discord](https://discord.gg/bMBhb7w) room.
Important event announcements will be made on both Discord “general” channel and the [dedicated GitHub issue](https://github.com/nilearn/nilearn/issues/2739).

Although we welcome any and all questions on Discord,
for transparency and continuity of development after the event we will encourage all technical discussions to be documented on GitHub on either the
[nilearn](https://github.com/nilearn/nilearn) or [nibabel](https://github.com/nipy/nibabel) issue board, as appropriate.

### Daily organization

To say hello and organize work, we will have two introductory meetings on Wednesday at 9:30a CET and 9:30a EDT.
To share progress we will have another common session on Thursday 5:00p CET, 11a EDT.

You can find all relevant events in [the sprint calendar](https://calendar.google.com/calendar/b/3?cid=bmlsZWFybi5ldmVudHNAZ21haWwuY29t).
We welcome all contributors to schedule discussions for specific topics of interest.
These sessions should be announced with at least a few hours notice on both the GitHub issue and Discord room.

We will consider the sprint hours to run roughly from 9:00a CET to 6:00p EDT.
We do not expect anyone to attend all (or most !) of that time !
These are just the times when at least one current Nilearn core developer will be online to help with general questions and point towards available resources.

### An introduction to Discord

Discord is a chat platform that allows for both voice and text communication. If you are new to discord, we encourage you to read the quick introduction guide [here](https://support.discord.com/hc/en-us/articles/360045138571-Beginner-s-Guide-to-Discord), which explains the basics of Discord.

In the left panel, you can modify settings at the top (notifications, privacy…) and see all available channels below (text and voice). In the right panel you can see who is connected and contact them directly.

The general channels will be for welcoming, general announcements, and asking general questions.
A break channel is dedicated to sharing breaks together. You can create other channels to accommodate specific needs (for example, setting up your development environment).

Notifications can be a bit overwhelming on Discord, don’t hesitate to filter them out while working (muting channels with right-click or setting your account on Do Not Disturb).

## Setting up a development environment

The first step is obviously to have Python installed and working. We recommand having a version of Python higher or equal to 3.6. If you don't have Python installed already, you can get it on the [official website](https://www.python.org/downloads/).

### Install Nilearn and Nibabel

Once you have Python installed correclty on your machine, you need to install [Nilearn](https://github.com/nilearn/nilearn),  [Nibabel](https://github.com/nipy/nibabel) and their dependencies. Both Nilearn and Nibabel are available through Pypi ([Nilearn link](https://pypi.org/project/nilearn/), [Nibabel link](https://pypi.org/project/nibabel/)) such that you should be able to pip-install them.

Actually, since Nibabel is a dependency of Nilearn, you can simply install everything you need by running:

```bash
$ pip install nilearn
```

Note that you may want to create a virtual environnement using for example [anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). More information for installing nilearn is available on the [online documentation](https://nilearn.github.io/introduction.html#installation).

### Checkout Nilearn contributing docs

Once you have everything installed and working, you can checkout the nilearn [contributing documentation](http://nilearn.github.io/development.html) which explains how to get set up to make contributions to the project.

### Browse selected issues

Finally, when you are ready to contribute, you can browse the issues we have selected specifically for the sprint on the following project boards:

- [Nilearn project board](https://github.com/nilearn/nilearn/projects/6)
- [Nibabel project board](https://github.com/nipy/nibabel/projects/1).