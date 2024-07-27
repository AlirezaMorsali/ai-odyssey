from lighter import Value, draw_dot

x1 = Value(3.0)
x2 = Value(6.0)

w1 = Value(10.0)
w2 = Value(23.0)
b = Value(0.3)


y = ((x1 * w1) + (x2 * w2)) + b

draw_dot(y)
print(f"{y=}")
print(f"{b=}")


pass
