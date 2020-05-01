---
layout: post
title: "Ubuntu16.04下更改Teamviewer ID"
subtitle: 'Teamviewr 破解'
author: "WenlSun"
header-img: "/img/post-bugs.jpg"
header-style: text
tags:
  - 远程控制
---

# Teamviewr 疑似商业用途  

![1561606363415](/img/teamviewr/fig1.png)

# 解决方案

## 通过更改MAC地址来更改Teamviewr的ID

1. 安装`macchanger`, 使用下面命令,一路默认即可.

   ```sh
   sudo apt-get install macchanger
   ```

2. 使用`ifconfig` 命令查看以太网 `enp5s0`和无线网 `wlp4s0`的名字以及`MAC`地址 `HWaddr`, 并对`MAC`地址进行备份:

   ```sh
   ifconfig
   ```

   ![1561607748206](/img/teamviewr/fig4.png)

3. 安装`teamviewr`

   + [官网](https://www.teamviewer.com/zhcn/download/linux/)下载`Ubuntu16.04`的`deb`安装文件
   + 运行以下命令进行安装

   ```sh
   sudo dpkg -i teamviewer_14.3.4730_amd64.deb 
   # 如果安装出错,是因为缺少依赖,使用下面的命令安装依赖,之后重新安装teamviewr
   sudo apt-get insatll -f
   ```

4. 更改`Teamviewer`的`ID`

   ```sh
   sudo teamviewer --daemon stop
   sudo ifconfig enp5s0 down
   sudo ifconfig enp5s0(此处是自己电脑以太网的名称) hw ether 00:11:22:33:44:55(此处是新的MAC地址)
   sudo ifconfig enp5s0 up
   sudo teamviewer --daemon start
   ```

   ![1561607153422](/img/teamviewr/fig.png)

5. 重启`Teamviewr` ,amazing!!!, ID已经更改了,从此再也不用担心五分钟限制啦!!!![1561607244621](/img/teamviewr/fig3.png)

