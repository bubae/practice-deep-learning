from theano import tensor as T

x = T.scalar()

y = T.scalar()

z = x + y

w = z * x

a = T.sqrt(w)

b = T.exp(a)

c = a ** b

d = T.log(c)

print x

print y

print z

print w

print a

print b

print c

print d