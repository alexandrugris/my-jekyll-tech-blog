---
layout: post
title:  "Welcome to Jekyll!"
date:   2017-01-26 22:45:16 +0200
categories: jekyll update
---
I was looking for a simple blogging platform that provided for my following needs.

 - Decouple content and formatting and be able to store it on cheap / free hosts with minimal hassle (e.g. github pages).
 - Total control over my content and not to be tied in to a blog provider (like Blogspot or Wordpress).
 - To be able to edit my content easily, just like I would take notes for myself.
 - To move my blog to anywhere I want and be am able to change formatting the way I want.

Writing content in Markdown seemed like a good, simple idea. Easy to write, total control over content, no complex editors, no vendor lock-in. 
Here is a screenshot of my editing experience in Visual Studio Code:

![Edit experience in Visual Studio Code]({{site.url}}/assets/jekyll_editing_experience.png)

As a bonus, Jekyll also offers powerful support for code snippets. Being a technical blog, this was just great.

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: http://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
