

@[TOC](文章目录)

---




# 一、本地控制
##  1.在一个目录下建立仓库

```cpp
git init
```
###  删除仓库

```cpp
rm -rf .git
```

##  2.版本控制
###  保存到暂存区
保存某个文件的修改到暂存区
```cpp
git add <filename>
```
全部保存到暂存区

```cpp
git add .
```
###  撤销暂存区文件
撤销某一个暂存区文件
```cpp
git reset HEAD <filename>
```
撤销所有暂存区文件

```cpp
git reset HEAD 
```
###  提交到版本库

```cpp
git commit -m "附加信息"
```
###  回退
#### 回退单一文件到上个版本

```cpp
git checkout -- <filename>
```
####  回退到上一个版本

```cpp
git reset --hard HEAD~1
```
####  回退到上n个版本
```cpp
git reset --hard HEAD~n
```
###  删除未跟踪的文件

```cpp
# 删除 untracked files
git clean -f
 
# 连 untracked 的目录也一起删掉
git clean -fd
 
# 连 gitignore 的untrack 文件/目录也一起删掉 （慎用，一般这个是用来删掉编译出来的 .o之类的文件用的）
git clean -xfd
 
# 在用上述 git clean 前，墙裂建议加上 -n 参数来先看看会删掉哪些文件，防止重要文件被误删
git clean -nxfd
git clean -nf
git clean -nfd
```
###  删除已跟踪的文件

```cpp
git rm <filename>
```

###  查看当前仓库状态

```cpp
git status
```
###  查看操作日志

```cpp
git reflog
```
###  查看此分支的提交日志

```cpp
git log
```

 - **经过回退后，使用"git log"就看不到回退前的版本号了，但使用"git reflog"就可以查看**
##  3.分支管理
###  创建分支

```cpp
git branch <name of local branch>
```
###  切换分支

```cpp
git switch <name of local branch>
```
###  查看分支

```cpp
git branch
```
###  合并分支（可能出现冲突，如两个分支都修改了同一文件）

```cpp
git merge <name of local branch>
git merge --no-ff -m "附加信息" <name of local branch>
```
###  删除分支

```cpp
git branch -d <name>
git branch -D <name>#强行删除，慎用
```

# 二、远程控制
## 1.获取本设备的ssh密钥

```cpp
ssh-keygen -t rsa -C "email@*.com"
```


## 2.与远程库建立或删除连接
###  建立连接
```cpp
git romote add <name of the remote repository> <SSH offered by the website>
```
###  删除连接

```cpp
git remote rm <name of the remote repository>
```

##  3.绑定此分支到远程库的某一分支

```cpp
git branch --set-upstream-to <name of local branch> <name of the remote repository >/<name of remote branch>
```
##  4.提交到远程库的某一分支

```cpp
git push <name of the remote repository> <name of local branch>
```

 - **如果远程库没有对应名字的分支，那么会在远程库自动创建一个同名分支**
 - **如果不给定\<name of local branch>参数，那么会按照git branch --set-upstream-to绑定的关系进行推送。如果未绑定，则推送失败**

##  5.标签管理
###  打标签

```cpp
git tag <name of tag> <version>
git tag -a <name of tag> -m "附加信息" <version> #带说明信息的标签
```

###  删除标签

```cpp
git tag -d <name of tag>
```

###  查看某一标签

```cpp
git show <name of tag>
```

###  查看所有标签

```cpp
git tag
```

###  推送标签到远程库

```cpp
git push <name of the remote repository> <name of tag>
git push <name of the remote repository> --tags #一次性推送所有标签
```

###  删除远程标签

```cpp
git push <name of the remote repository>:refs/tags/<name of tag>
```

#  三、多人协作
##  1.远古bug修复流程

 1. 保存此分支的工作现场
```cpp
git stash
```

 2. 切换到有bug的分支，从这个有bug的分支创建一个新分支修复bug，然后合并，**保存好合并时给出的版本号**
 3. **用这个版本号**，对所有存在这个bug的分支做一次相同的提交
```cpp
git cherry-pick <version>
```

 4. 回到工作现场，查看贮藏列表并恢复
```cpp
git stash list #列出贮藏列表
git stash apply stash@{n} #根据序号n恢复到指定工作现场
git stash drop stash@{n}#根据序号n删除指定工作现场
```
##  2.版本落后于远程库而推送失败

 1. 先用git branch --set-upstream-to 绑定到指定远程分支
```cpp
git branch --set-upstream-to <name of local branch> <name of the remote repository >/<name of remote branch>
```

 2. 用git pull指令：抓取所绑定分支的最新版本并尝试merge。若出现冲突要手动解决

```cpp
git pull
```

 3. 最后提交并推送即可
 

```cpp
git commit -m "附加信息"
git push <name of the remote repository>
```

