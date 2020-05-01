---
layout: post
title: "使用 github.io 创建自己的博客"
subtitle: 'Create your own blog with github.io'
author: "WenlSun"
header-img: "/img/post-bugs.jpg"
header-style: text
tags:
  - 笔记
  - github.io
---

### 创建仓库

首先需要有自己的 [github](https://github.com/) 账号，然后创建一个新仓库，这个新的仓库就是存放你的博客的地方。注意：仓库的名字应该为`用户名.github.io`，其中`用户名`为你的 github 的账户名称。例如：你的 github 账户名为：`wenliangsun`,则创建的仓库名称为`wenliangsun.github.io`。

![](/img/set-github/post-set-github-io-create-repo.png)

![](/img/set-github/post-set-github-io-create-repo2.png)

### 设置参数

> 创建完仓库后，进入新创建的仓库，找到设置，进入设置界面。

![](/img/set-github/post-set-github-io-set-repo.png)

![](/img/set-github/post-set-github-io-set-repo2.png)

![](/img/set-github/post-set-github-io-set-repo3.png)

> 至此，在浏览器中输入`用户名.github.io`就可以看到效果。



### 选择一款`jekyll` 的主题

`github.io`默认采用`jekyll`作为建站工具，[Jekyll](https://jekyllrb.com/)是一款当前火热的开源的静态网站建站工具，拥有非常庞大的使用群里和社区，其[Github](https://github.com/jekyll/jekyll)截止本文，已经有超过3W+的star，拥有丰富的插件，丰富的主题，并且有无数的人已经帮你早出了无数的轮子可供参考。`Jekyll`自身的强大功能已经足够你打造自己心仪的静态网站（这里注意的是`静态网站`，`Jekyll`没有任何的后台数据库），然而前提是你自己还是得有一定的前端功底，而为了不至于长的太难看，你还得有一定的设计能力。然则，正如刚刚反复强调的，`Jekyll`已经有一个非常庞大的社区，这就意味着，你完全可以借鉴别人已经造好的轮子，放在`Jekyll`这里，咱们应该成为主体（Theme）比较合适。本文推荐国内用户可以考虑一款[国人开发的主题](https://github.com/Huxpro/huxpro.github.io)。



#### clone 主题(`jekyll`)

```git
git clone https://github.com/Huxpro/huxpro.github.io
```

#### clone 自己的repo到本地

```
git git@github.com:wenliangsun/wenliangsun.github.io.git
```

#### 将huxpro.github.io中的文件(除.git外)复制到自己的github.io的文件夹中

![](/img/set-github/post-set-github-io-select-theme.png)

#### 修改`_config.yml`配置文件

根据自己的信息修改配置文件

![](/img/set-github/post-set-github-io-select-theme2.png)

#### 编写和发布博客

+ 博客文件都是要求放在`_posts`目录下面，同时对博文的文件名有[严格的规定](https://jekyllrb.com/docs/posts/#creating-post-files)，必须保持格式`YEAR-MONTH-DAY-title.MARKUP`，通常情况下，咱们采用推荐的`Markdown`撰写博文，基于该格式。

+ git 提交自己的博文

  ```
  git add --all
  git commit -m "add file"
  git push -u origin master
  ```

+ 在浏览器中输入`用户名.github.io`即可查看自己的博文

#### 配置`github page` 使得支持`Letax`公式

在`_include/head.html`文件中的`<head>...</head>`中添加以下代码

```html
<head>
    ...
    <script type="text/x-mathjax-config"> 
   		MathJax.Hub.Config({ TeX: { equationNumbers: { autoNumber: "all" } } }); 
   	</script>
    <script type="text/x-mathjax-config">
    	MathJax.Hub.Config({tex2jax: {
             inlineMath: [ ['$','$'], ["\\(","\\)"] ],
             processEscapes: true
           }
         });
    </script>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript">
    </script>
    ...
</head>
```



### `jekyll` 是如何工作的

在`jekyll`解析你的网站之前，需要确保网站有以下目录结构：

```
|-- _config.yml
|-- _includes
|-- _layouts
|   |-- default.html
|   |-- post.html
|-- _posts
|   |-- 20011-10-25-open-source-is-good.html
|   |-- 20011-04-26-hello-world.html
|-- _site
|-- index.html
|-- images
   |-- css
       |-- style.css
   |-- javascripts
```

> + `_config.yml`：保存配置，该配置将影响`jekyll`构造网站的各种行为。

> + `_includes`：改目录下的文件可以用来作为公共的内容被其他文章引用，就和c语言include头文件的机制是一样的，`jekyll`在解析时会对`{ % include file.ext %}`标记扩展成对应的在`_includes`文件夹中的文件。

> + `_layouts`：该目录下的文件作文主要的模板文件。

> + `_posts`：文章或网页应当放在这个目录中，但需要注意的是，文章的文件名必须是`YYYY-MM-DD-title`。

> + `_site`：这是`jekyll`默认的转化结果存放的目录。

> + `images`：这个目录没有强制的要求，主要目的是存放你的资源文件，图片、样式表、脚本等。

以上就是大概的简介，可以在`_config.yml`文件中修改网站以及自己的个人信息，`_posts`文件夹放你的博文，切记文件名格式，更多详细的语法上面贴出来的`jekyll`的官方博客以及`github`可以参考。



### `jekyll` 安装本地服务

如果每次改动都需要提交到远端仓库才能看到效果的话，那就太麻烦了，有时候我们需要在本地测试好了，才提交上去，这就需要我们安装`jekyll`本地服务。

未完，待填。。。。。









#### 参考文献

[1] https://www.jianshu.com/p/fabb01427203

[2] https://keysaim.github.io/post/blog/2017-08-15-how-to-setup-your-github-io-blog/

