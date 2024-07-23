a = T.tensor([3.0])
a.requires_grad = True
b = T.tensor([2.0])
b.requires_grad = True

y = a * b
print(f"{y=}")
y.backward()

print(f"{a.grad=}")
print(f"{b.grad=}")
