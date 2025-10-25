import matplotlib.pyplot as plt
import numpy as np

# Данные из лога
epochs = [0.08, 0.17, 0.25, 0.33, 0.42, 0.5, 0.58, 0.67, 0.75, 0.83, 0.92, 1.0, 1.08, 1.17, 1.25, 1.33, 1.42, 1.5, 1.58, 1.67, 1.75, 1.83, 1.92, 2.0, 2.08, 2.17, 2.25, 2.33, 2.42, 2.5, 2.58, 2.67, 2.75, 2.83, 2.92, 3.0, 3.08, 3.17, 3.25, 3.33, 3.42, 3.5, 3.58, 3.67, 3.75, 3.83, 3.92, 4.0, 4.08, 4.17, 4.25, 4.33, 4.42, 4.5, 4.58, 4.67, 4.75, 4.83, 4.92, 5.0, 5.08, 5.17, 5.25, 5.33, 5.42, 5.5, 5.58, 5.67, 5.75, 5.83, 5.92, 6.0, 6.08, 6.17, 6.25, 6.33, 6.42, 6.5, 6.58, 6.67, 6.75, 6.83, 6.92, 7.0]

losses = [19.6828, 17.1354, 14.2693, 11.3293, 8.1648, 4.6447, 1.4644, 0.3427, 0.2882, 0.2835, 0.2972, 0.2134, 0.0991, 0.1844, 0.1077, 0.1062, 0.104, 0.0723, 0.0791, 0.1118, 0.1025, 0.0967, 0.0885, 0.1027, 0.0834, 0.0924, 0.0837, 0.087, 0.0675, 0.0728, 0.0912, 0.0863, 0.0721, 0.0576, 0.0935, 0.0803, 0.0844, 0.0738, 0.0809, 0.0795, 0.0652, 0.0925, 0.0762, 0.0775, 0.082, 0.0757, 0.0776, 0.074, 0.06, 0.0844, 0.0768, 0.0721, 0.0885, 0.0764, 0.071, 0.0799, 0.0681, 0.068, 0.0833, 0.0636, 0.0679, 0.0843, 0.0786, 0.0739, 0.067, 0.0753, 0.0777, 0.0706, 0.0839, 0.0774, 0.079, 0.0625, 0.0783, 0.0662, 0.0782, 0.0736, 0.0639, 0.0794, 0.0684, 0.0617, 0.072, 0.0826, 0.0785, 0.0756]

grad_norms = [15.6877, 10.3514, 15.2839, 14.9945, 15.1026, 14.7452, 9.5857, 1.8885, 1.1460, 1.2858, 3.8070, 4.7551, 0.5584, 1.3811, 0.6227, 0.6241, 1.3980, 0.5169, 0.5804, 0.4518, 0.5495, 0.7949, 0.4415, 0.5113, 0.4417, 0.3657, 0.7395, 0.2186, 0.3977, 0.3371, 0.2231, 0.4284, 0.2961, 0.1688, 0.3245, 0.2774, 0.2010, 0.2635, 0.4187, 0.2390, 0.2762, 0.2125, 0.2182, 0.2028, 0.3097, 0.2967, 0.2121, 0.3326, 0.3246, 0.2039, 0.3165, 0.3683, 0.2529, 0.2532, 0.2555, 0.2397, 0.1886, 0.1905, 0.3700, 0.1846, 0.1803, 0.1977, 0.2567, 0.5811, 0.2176, 0.1812, 0.1830, 0.1771, 0.2193, 0.1802, 0.2063, 0.1700, 0.2086, 0.2118, 0.1974, 0.1772, 0.2121, 0.1872, 0.1752, 0.1765, 0.2717, 0.1974, 0.1788, 0.2039]

learning_rates = [0.0001, 9.88095e-05, 9.76190e-05, 9.64286e-05, 9.52381e-05, 9.40476e-05, 9.28571e-05, 9.16667e-05, 9.04762e-05, 8.92857e-05, 8.80952e-05, 8.69048e-05, 8.57143e-05, 8.45238e-05, 8.33333e-05, 8.21429e-05, 8.09524e-05, 7.97619e-05, 7.85714e-05, 7.73810e-05, 7.61905e-05, 7.50000e-05, 7.38095e-05, 7.26190e-05, 7.14286e-05, 7.02381e-05, 6.90476e-05, 6.78571e-05, 6.66667e-05, 6.54762e-05, 6.42857e-05, 6.30952e-05, 6.19048e-05, 6.07143e-05, 5.95238e-05, 5.83333e-05, 5.71429e-05, 5.59524e-05, 5.47619e-05, 5.35714e-05, 5.23810e-05, 5.11905e-05, 5e-05, 4.88095e-05, 4.76190e-05, 4.64286e-05, 4.52381e-05, 4.40476e-05, 4.28571e-05, 4.16667e-05, 4.04762e-05, 3.92857e-05, 3.80952e-05, 3.69048e-05, 3.57143e-05, 3.45238e-05, 3.33333e-05, 3.21429e-05, 3.09524e-05, 2.97619e-05, 2.85714e-05, 2.73810e-05, 2.61905e-05, 2.5e-05, 2.38095e-05, 2.26190e-05, 2.14286e-05, 2.02381e-05, 1.90476e-05, 1.78571e-05, 1.66667e-05, 1.54762e-05, 1.42857e-05, 1.30952e-05, 1.19048e-05, 1.07143e-05, 9.52381e-06, 8.33333e-06, 7.14286e-06, 5.95238e-06, 4.76190e-06, 3.57143e-06, 2.38095e-06, 1.19048e-06]

plt.figure(figsize=(15, 12))

# График 1: Loss (логарифмическая шкала)
plt.subplot(3, 1, 1)
plt.plot(epochs, losses, 'b-', linewidth=2, alpha=0.8)
plt.title('Training Loss - Full Scale (Log)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.yscale('log')
# Добавляем аннотации
improvement = losses[0] / losses[-1]
plt.annotate(f'Улучшение: {improvement:.0f}x',
             xy=(0.65, 0.15), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))

# График 2: Loss после эпохи 0.7 (линейная шкала) - детальный вид
plt.subplot(3, 1, 2)
# Находим индекс, где epoch > 0.7
mask = np.array(epochs) > 0.7
epochs_filtered = np.array(epochs)[mask]
losses_filtered = np.array(losses)[mask]

plt.plot(epochs_filtered, losses_filtered, 'r-', linewidth=2, alpha=0.8)
plt.title('Training Loss - Detailed View (After Epoch 0.7)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.ylim(0.05, 0.12)
# Горизонтальная линия для среднего значения
mean_loss = np.mean(losses_filtered)
plt.axhline(y=mean_loss, color='green', linestyle='--', alpha=0.7, label=f'Среднее: {mean_loss:.3f}')
plt.legend()

# График 3: Gradient Norm
plt.subplot(3, 1, 3)
plt.plot(epochs, grad_norms, 'g-', linewidth=2, alpha=0.8)
plt.title('Gradient Norm during Training', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Gradient Norm')
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()

# Сохраняем график
plt.savefig('training_analysis_v3.png', dpi=300, bbox_inches='tight')
print("График сохранен как 'training_analysis_v3.png'")

# Дополнительный график: Learning Rate
plt.figure(figsize=(12, 4))
plt.plot(epochs, learning_rates, 'purple', linewidth=2)
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.savefig('learning_rate_schedule_v3.png', dpi=300, bbox_inches='tight')
print("График learning rate сохранен как 'learning_rate_schedule_v3.png'")

# Выводим статистику
print(f"\n=== СТАТИСТИКА ОБУЧЕНИЯ ===")
print(f"Параметры модели:")
print(f"  Всего параметров: 1,193,726,720")
print(f"  Обучаемых параметров: 419,696,640")
print(f"  Процент обучаемых: 35.16%")

print(f"\nМетрики обучения:")
print(f"Начальный loss: {losses[0]:.4f}")
print(f"Финальный loss: {losses[-1]:.4f}")
print(f"Улучшение: {improvement:.0f}x")
print(f"Минимальный loss: {min(losses):.4f}")
print(f"Всего эпох: {epochs[-1]:.1f}")
print(f"Количество шагов: {len(losses)}")
print(f"Общее время обучения: 420.51 секунд ({420.51/60:.1f} минут)")
print(f"Средний loss: 0.9994")

# Анализ сходимости после быстрого падения
final_losses = losses[7:]  # После эпохи 0.67
loss_std = np.std(final_losses)
mean_final_loss = np.mean(final_losses)

print(f"\n=== АНАЛИЗ СХОДИМОСТИ ===")
print(f"Средний loss после эпохи 0.67: {mean_final_loss:.4f}")
print(f"Стандартное отклонение: {loss_std:.4f}")
print(f"Диапазон: {max(final_losses):.4f} - {min(final_losses):.4f}")

if loss_std < 0.02:
    convergence_status = "✅ Отличная сходимость"
elif loss_std < 0.05:
    convergence_status = "✅ Хорошая сходимость"
else:
    convergence_status = "⚠️ Умеренная сходимость"

print(f"Статус: {convergence_status}")

# Анализ скорости обучения
print(f"\n=== АНАЛИЗ СКОРОСТИ ОБУЧЕНИЯ ===")
print(f"Скорость обучения: 0.2 шага/сек")
print(f"Время на эпоху: ~60 секунд")
print(f"Эффективность: Очень быстрое начальное обучение")

plt.show()