# Setup
## Firmware
该项目的硬件平台为树莓派4b 4G版本。

在树莓派上安装了64位最新版基于bulleyes版Debian的树莓派系统，系统用户信息如下：

- Hostname: raspberrypi
- Username: pi
- Password: raspberry

## 树莓派系统登陆、操作方式
1. 使用显示器、键鼠
2. ssh
  通过ssh连接需要注意以下事项，加上`-X`选项才能通过X11 Forwarding将图形界面显示出来，这样才能运行本项目。

  另外，ssh连接需要在同一局域网才能连接，如果一开始没有显示器等外设，就没有办法操作连接Wifi，这样就没法连到同一个局域网了。这个问题可以通过将路由器（或者手机热点）的账号密码设置为以下事先保存的Wifi：

  > 账号：407
  > 密码：407407407

  设置完后，树莓派会自动连接这个Wifi。


# Reference
[开源工业缺陷数据集汇总，持续更新中（已更新28个） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/195699093)
