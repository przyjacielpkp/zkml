
# zkml

## supported ops

We support the ops you can more or less support natively with snarks.
That is:
 - Add
 - Mul
 - LessThan  (Q: is this expensive?)
 - Constant
 - Recip
 - MaxReduce
 - SumReduce

Should/Can we support:
 - Sqrt? possible
 - Mod?  possible
 - Exp?  ?

We don't support nonlinear functions:
 - Sin, Tan, Log2

This allows us to implement i.e. fully connected Relu nets, convolutions etc.
