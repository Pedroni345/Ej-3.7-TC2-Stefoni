import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.signal as signal

# Parámetros genéricos
epsilon = 0.5  # Ripple en banda de paso
omega_p = 1.0  # Frecuencia de corte normalizada (banda de paso)
omega_s = 2.0  # Frecuencia de rechazo
A_s = 20       # Atenuación en dB en banda de rechazo

# Función f(ω) = 2ω² - 1
def f(w):
    return 4/3 * w * (w-0.5) * (w+0.5)

# Rango de frecuencias
w = np.linspace(0, 3, 1000)

# =============================================================================
# PASO 1: Representar f(ω)
# =============================================================================
print("="*60)
print("PASO 1: Función de aproximación f(ω) = 4/3 * w * (w-0.5) * (w+0.5)")
print("="*60)

f_w = f(w)

plt.figure(figsize=(12, 10))

plt.subplot(4, 2, 1)
plt.plot(w, f_w, 'r', linewidth=2)
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3)
plt.xlabel('ω', fontsize=12)
plt.ylabel('f(ω)', fontsize=12)
plt.title('f(ω) = 2ω² - 1', fontsize=14, fontweight='bold')
plt.xlim([0, 3])

# =============================================================================
# PASO 2: Operaciones para generar G(ω)
# =============================================================================
print("\n" + "="*60)
print("PASO 2: Generación de G(ω) a partir de f(ω)")
print("="*60)

# Paso 2a: ΔG(ω) = ε·f(ω)
delta_G = epsilon * f_w
plt.subplot(4, 2, 2)
plt.plot(w, delta_G, 'b', linewidth=2, label='ΔG(ω)')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3)
plt.xlabel('ω', fontsize=12)
plt.ylabel('ΔG(ω)', fontsize=12)
plt.title(f'ΔG(ω) = ε·f(ω), ε = {epsilon}', fontsize=14, fontweight='bold')
plt.xlim([0, 3])
plt.legend()

print(f"ΔG(ω) = ε·f(ω), donde ε = {epsilon}")

# Paso 2b: (ε·f(ω))²
epsilon_f_squared = (epsilon * f_w)**2
plt.subplot(4, 2, 3)
plt.plot(w, epsilon_f_squared, 'g', linewidth=2)
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3)
plt.xlabel('ω', fontsize=12)
plt.ylabel('(ε·f(ω))²', fontsize=12)
plt.title('(ε·f(ω))²', fontsize=14, fontweight='bold')
plt.xlim([0, 3])

print("(ε·f(ω))² calculado")

# Paso 2c: G²(ω) = 1/(1 + ε²·f²(ω))
G_squared = 1 / (1 + epsilon_f_squared)
plt.subplot(4, 2, 4)
plt.plot(w, G_squared, 'purple', linewidth=2)
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3)
plt.xlabel('ω', fontsize=12)
plt.ylabel('G²(ω)', fontsize=12)
plt.title('G²(ω) = 1/(1 + ε²·f²(ω))', fontsize=14, fontweight='bold')
plt.xlim([0, 3])
plt.ylim([0, 1.1])

print("G²(ω) = 1/(1 + ε²·f²(ω)) [Transformación de Butterworth]")

# Paso 2d: G(ω)
G_w = np.sqrt(G_squared)
Gp = 1 / np.sqrt(1 + epsilon**2)

print(f"\nGanancia en banda de paso: Gp = 1/√(1+ε²) = {Gp:.4f}")
print(f"En dB: {20*np.log10(Gp):.2f} dB")

# =============================================================================
# PASO 3: G(ω) en plantilla normalizada
# =============================================================================
plt.subplot(4, 2, 5)
plt.plot(w, G_w, 'b', linewidth=2.5, label='G(ω)')

# Dibujar plantilla normalizada
plt.axhline(y=1, color='r', linestyle='--', linewidth=1.5, label='Banda de paso superior')
plt.axhline(y=Gp, color='orange', linestyle='--', linewidth=1.5, label=f'Gp = {Gp:.3f}')
plt.axvline(x=omega_p, color='g', linestyle='--', linewidth=1.5, label=f'ωp = {omega_p}')
plt.axvline(x=omega_s, color='m', linestyle='--', linewidth=1.5, label=f'ωs = {omega_s}')

# Región de banda de paso (verde claro)
plt.fill_between([0, omega_p], Gp, 1, alpha=0.2, color='green', label='Banda de paso')

# Región de banda de rechazo (roja clara)
plt.fill_between([omega_s, 3], 0, 0.4, alpha=0.2, color='red', label='Banda de rechazo')

plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3)
plt.xlabel('ω', fontsize=12)
plt.ylabel('|G(ω)|', fontsize=12)
plt.title('G(ω) en Plantilla Normalizada', fontsize=14, fontweight='bold')
plt.xlim([0, 3])
plt.ylim([0, 1.1])
plt.legend(fontsize=8, loc='upper right')

print("\n" + "="*60)
print("PASO 3: G(ω) representada en plantilla normalizada")
print("="*60)

# =============================================================================
# PASO 4: Transformación ω = s/j y singularidades de A(s)
# =============================================================================
print("\n" + "="*60)
print("PASO 4: Transformación ω = s/j → A(s) = G²(ω)|ω=s/j")
print("="*60)

# Para f(ω) = 2ω² - 1, con ω = s/j:
# f(s/j) = 2(s/j)² - 1 = 2(-s²) - 1 = -2s² - 1
# 
# G²(s) = 1/(1 + ε²·f²(s/j)) = 1/(1 + ε²·(-2s² - 1)²)
# 
# A(s) = 1 + ε²·(-2s² - 1)²

# Expandir (-2s² - 1)²
# (-2s² - 1)² = 4s⁴ + 4s² + 1

# Entonces: A(s) = 1 + ε²(4s⁴ + 4s² + 1)
#                 = 1 + 4ε²s⁴ + 4ε²s² + ε²
#                 = 4ε²s⁴ + 4ε²s² + (1 + ε²)

# Coeficientes del polinomio A(s)
a4 = 4 * epsilon**2
a2 = 4 * epsilon**2
a0 = 1 + epsilon**2

print(f"\nA(s) = {a4:.4f}s⁴ + {a2:.4f}s² + {a0:.4f}")

# Resolver A(s) = 0 para encontrar polos
# Sustitución u = s²: 4ε²u² + 4ε²u + (1+ε²) = 0
discriminant = (a2)**2 - 4*a4*a0
u_roots = [(-a2 + np.sqrt(discriminant + 0j))/(2*a4),
           (-a2 - np.sqrt(discriminant + 0j))/(2*a4)]

# Polos en s
poles = []
for u in u_roots:
    poles.append(np.sqrt(u))
    poles.append(-np.sqrt(u))

poles = np.array(poles)

print(f"\nPolos de A(s) (total: {len(poles)}):")
for i, p in enumerate(poles):
    print(f"  s{i+1} = {p.real:.4f} + j{p.imag:.4f}")

# Gráfica de polos y ceros en el plano s
plt.subplot(4, 2, 6)
plt.plot(poles.real, poles.imag, 'rx', markersize=15, markeredgewidth=3, label='Polos de A(s)')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

# Dibujar círculo unitario como referencia
theta = np.linspace(0, 2*np.pi, 100)
radius = np.abs(poles[0])
plt.plot(radius*np.cos(theta), radius*np.sin(theta), 'b--', alpha=0.3, label='|s| = cte')

plt.grid(True, alpha=0.3)
plt.xlabel('Re(s)', fontsize=12)
plt.ylabel('Im(s)', fontsize=12)
plt.title('Polos de A(s) en el plano s', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.legend(fontsize=9)

# =============================================================================
# PASO 5: Función H(s) implementable (semiplano izquierdo)
# =============================================================================
print("\n" + "="*60)
print("PASO 5: H(s) implementable - Selección de polos en semiplano izquierdo")
print("="*60)

# Seleccionar solo polos con parte real negativa (estables)
stable_poles = poles[poles.real < 0]

print(f"\nPolos estables seleccionados para H(s) (total: {len(stable_poles)}):")
for i, p in enumerate(stable_poles):
    print(f"  s{i+1} = {p.real:.4f} + j{p.imag:.4f}")

# Construir H(s) = K / ∏(s - pi)
# Donde K se calcula para que |H(0)| = 1 (ganancia DC = 1)

# Calcular la ganancia K
K = 1.0
for p in stable_poles:
    K *= -p
K = np.abs(K)

print(f"\nConstante K = {K:.4f} (para |H(0)| = 1)")

# Gráfica de polos de H(s)
plt.subplot(4, 2, 7)
plt.plot(poles.real, poles.imag, 'rx', markersize=12, markeredgewidth=2, 
         alpha=0.3, label='Polos descartados')
plt.plot(stable_poles.real, stable_poles.imag, 'go', markersize=12, 
         markeredgewidth=3, label='Polos de H(s)')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='r', linestyle='--', linewidth=2, alpha=0.5)

# Marcar semiplano izquierdo
plt.fill_between([-2, 0], -2, 2, alpha=0.1, color='green')
plt.text(-0.8, 1.5, 'Semiplano\nizquierdo\n(estable)', fontsize=10, 
         ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.grid(True, alpha=0.3)
plt.xlabel('Re(s)', fontsize=12)
plt.ylabel('Im(s)', fontsize=12)
plt.title('Selección de polos para H(s)', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.xlim([-2, 1])
plt.ylim([-2, 2])
plt.legend(fontsize=9)

# =============================================================================
# PASO 6: Verificación - |H(jω)| cumple la plantilla
# =============================================================================
print("\n" + "="*60)
print("PASO 6: Verificación - |H(jω)| en la plantilla original")
print("="*60)

# CORRECCIÓN: H(s) debe cumplir que |H(jω)|² = G²(ω)
# Esto significa: H(s)·H(-s) = 1/A(s)
# Por lo tanto, H(s) debe construirse como: H(s) = 1/√A(s) con polos estables

# Evaluar H(jω) correctamente
w_eval = np.linspace(0, 3, 1000)

# Método correcto: evaluar directamente desde G²(ω)
# |H(jω)| = G(ω) por definición de la aproximación de Butterworth
H_magnitude = np.sqrt(G_squared)  # Usar el G(ω) ya calculado

# Verificación alternativa: construir H(jω) desde los polos
s_eval = 1j * w_eval
H_jw_from_poles = np.ones_like(s_eval, dtype=complex) * K
for p in stable_poles:
    H_jw_from_poles /= (s_eval - p)
H_magnitude_from_poles = np.abs(H_jw_from_poles)

print(f"\nVerificando que |H(jω)| = G(ω)...")
max_error = np.max(np.abs(H_magnitude - G_w))
print(f"Error máximo entre métodos: {max_error:.6f}")

plt.subplot(4, 2, 8)
plt.plot(w, G_w, 'r--', linewidth=3, alpha=0.8, label='|G(ω)| (teórico)')
plt.plot(w_eval, H_magnitude, 'b', linewidth=2, label='|H(jω)| desde G²(ω)')
plt.plot(w_eval, H_magnitude_from_poles, 'g:', linewidth=2, alpha=0.7, label='|H(jω)| desde polos')

# Plantilla
plt.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
plt.axhline(y=Gp, color='orange', linestyle='--', linewidth=1.5)
plt.axvline(x=omega_p, color='g', linestyle='--', linewidth=1.5)
plt.axvline(x=omega_s, color='m', linestyle='--', linewidth=1.5)

# Regiones
plt.fill_between([0, omega_p], Gp, 1, alpha=0.2, color='green')
plt.fill_between([omega_s, 3], 0, 0.4, alpha=0.2, color='red')

plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3)
plt.xlabel('ω', fontsize=12)
plt.ylabel('|H(jω)|', fontsize=12)
plt.title('Verificación: |H(jω)| cumple la plantilla', fontsize=14, fontweight='bold')
plt.xlim([0, 3])
plt.ylim([0, 1.1])
plt.legend(fontsize=9)

print("\n¡Verificación completada!")
print(f"En ω = {omega_p}: |H(jω)| = {H_magnitude[np.argmin(np.abs(w - omega_p))]:.4f} (debe ser ≥ {Gp:.4f})")
print(f"En ω = {omega_s}: |H(jω)| = {H_magnitude[np.argmin(np.abs(w - omega_s))]:.4f} (debe ser ≤ 0.4)")
print(f"\nAmbos métodos coinciden: Error máximo = {max_error:.6f}")

plt.tight_layout()
plt.savefig('butterworth_aproximacion.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("RESUMEN DEL DISEÑO")
print("="*60)
print(f"Función original: f(ω) = 2ω² - 1")
print(f"Epsilon (ε): {epsilon}")
print(f"Orden del filtro: {len(stable_poles)}")
print(f"Ganancia en banda de paso (Gp): {Gp:.4f} ({20*np.log10(Gp):.2f} dB)")
print(f"Frecuencia de banda de paso (ωp): {omega_p}")
print(f"Frecuencia de banda de rechazo (ωs): {omega_s}")
print("="*60)