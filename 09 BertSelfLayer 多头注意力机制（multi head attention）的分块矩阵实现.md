# 单Head
Embedding --> SelfAttn[Wq Wk Wv learnable] --> attn output

E: 1,T,d


E * Wq (d * dq) -> Q
E * Wk (d * dk)-> K
E * Wv (d * dv)-> V


softmax(Q * K.T / sqrt(dk)) ==> attention scores 纬度是T * T, 也体现了self attention

attn scores * V (T * dv)  ==> T * dv

dq 必须等于 dk，但是dv不做要求。


# 多head
W q|k|v 1|...

分块矩阵：
768 * (64 * 12)
[E * Wq1| E* Wq2 | ...| E * Wqn]  T * (n * dq)
[E * Wk1| E* Wk2 | ...| E * Wkn]  T * (n * dk)
[E * Wv1| E* Wv2 | ...| E * Wvn]  T * (n * dv)


但是，可以单独用一个大矩阵。
E * [Wq1|Wq2|...Wqn], KV一样。也是有Wq Wk Wv



[softmax(q1 * k1) /... * v1 | softmax(q2* k2)/.. * v2|...]  concat

最终输出 T * （n * dv）
