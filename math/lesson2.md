# 命题逻辑等值演算
## 等值式
### 等值式的定义
若等值式$A \leftrightarrow B$是一个命题公式，且$A$和$B$的真值表相同，则称$A \Leftrightarrow B$为等值式。
$\Leftrightarrow$表示等值式,是一个元语言定义符号，$\leftrightarrow$表示等价命题。
### 等值式的性质
$(A\rightarrow B)\rightarrow r \Leftrightarrow (A\wedge B)\rightarrow r $
#### 基本等值式
1. 交换律：$A\wedge B \Leftrightarrow B\wedge A$，$A\vee B \Leftrightarrow B\vee A$
1. 结合律：$A\wedge (B\wedge r) \Leftrightarrow (A\wedge B)\wedge r$，$A\vee (B\vee r) \Leftrightarrow (A\vee B)\vee r$
1. 分配律：$A\wedge (B\vee r) \Leftrightarrow (A\wedge B)\vee (A\wedge r)$，$A\vee (B\wedge r) \Leftrightarrow (A\vee B)\wedge (A\vee r)$
1. 吸收律：$A\wedge (A\vee B) \Leftrightarrow A$，$A\vee (A\wedge B) \Leftrightarrow A$被吸收的在括号里，括号内外符号不相同
1. 双重否定律：$\neg (\neg A) \Leftrightarrow A$
1. 幂等律：$A\wedge A \Leftrightarrow A$，$A\vee A \Leftrightarrow A$
1. 德摩根律：$\neg (A\wedge B) \Leftrightarrow \neg A\vee \neg B$ 并不是p或者q都成立->，$\neg (A\vee B) \Leftrightarrow \neg A\wedge \neg B$
1. 零律：$A\wedge F \Leftrightarrow F$，$A\vee T \Leftrightarrow T$ 零元：F，T
2. 同一律：$A\wedge T \Leftrightarrow A$，$A\vee F \Leftrightarrow A$
2. 排中律：$A\vee \neg A \Leftrightarrow T$
2. 矛盾律：$A\wedge \neg A \Leftrightarrow F$
1. 蕴含等值式：$A\rightarrow B \Leftrightarrow \neg A\vee B$
1. 假言易位式：$A\rightarrow B \Leftrightarrow \neg B\rightarrow \neg A$
1. 等价等值式：$A\leftrightarrow B \Leftrightarrow (A\rightarrow B)\wedge (B\rightarrow A)$
1. 等价否定等值式：$ A\rightarrow B \Leftrightarrow (\neg A\rightarrow \neg B)$
1. 归谬论：$(A\rightarrow B) \wedge (A\rightarrow \neg B) \Leftrightarrow \neg A$

## 对偶式
### 对偶式的定义

## 反演式
### 反演式的定义

# 等值演算与置换规则
## 等值演算
### 等值演算的定义

## 置换规则

一般先把蕴含规则去掉 （用了哪个定律需要写出来，置换律不用写）

# 吸取范式与合取范式
## 文字
命题变项及其否定的总称
## 简单合取式

## 简单析取式
## 吸取范式
是矛盾式当且仅当它的文字中都是相互矛盾的简单合取式