from sage.all import var

# Declare symbolic variables
t0, t1, t2 = var("t0 t1 t2")
delta_min, delta_a = var("delta_min delta_a")
R_A, R_B = var("R_A R_B")
C_1, C_2 = var("C_1 C_2")

# Step 1: Case (a) - First rise
t_o_1 = t0 + delta_a

# Step 2: Case (a, c)
T_1 = t1 - t_o_1
delta_ac = (-C_2 * R_B * (T_1 + delta_min)) / (C_1 * (R_A + R_B)) + delta_min
t_o_2 = t1 + delta_ac

# Step 3: Case (c, e)
T_2 = t2 - t_o_2
delta_ce = (-C_1 * (R_A + R_B) * (T_2 + delta_min)) / (C_2 * R_A) + delta_min
t_o_3 = t2 + delta_ce

# Print expressions without expansion
print("t_o_1 =", t_o_1)
print("t_o_2 =", t_o_2)
print("t_o_3 =", t_o_3)

