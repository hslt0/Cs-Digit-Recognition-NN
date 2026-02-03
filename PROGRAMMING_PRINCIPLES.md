# Принципи програмування в проєкті DigitRecognitionNN

Цей документ описує дотримання принципів програмування в коді проєкту з посиланнями на відповідні файли та рядки.

## 1. Single Responsibility Principle (SRP) - Принцип єдиної відповідальності

Кожен клас повинен мати лише одну причину для змін.

*   **Клас `Matrix`**: Відповідає виключно за математичні операції з матрицями (створення, додавання, множення, транспонування). Він не займається логікою нейромережі чи завантаженням даних.
    *   [DigitRecognitionNN/Models/Matrix.cs:5](DigitRecognitionNN/Models/Matrix.cs#L5)

*   **Клас `MathUtils`**: Містить лише допоміжні математичні функції, які не є частиною класу Matrix, такі як генерація випадкових ваг або обчислення похибки.
    *   [DigitRecognitionNN/Utils/MathUtils.cs:5](DigitRecognitionNN/Utils/MathUtils.cs#L5)

## 2. Open/Closed Principle (OCP) - Принцип відкритості/закритості

Програмні сутності повинні бути відкриті для розширення, але закриті для модифікації.

*   **Оператори в `Matrix`**: Ми можемо додавати нові операції над матрицями (наприклад, нові перевантаження операторів), не змінюючи існуючий код, який використовує ці матриці.
    *   [DigitRecognitionNN/Models/Matrix.cs:96](DigitRecognitionNN/Models/Matrix.cs#L96) (Оператор додавання)

## 3. Encapsulation - Інкапсуляція

Приховування внутрішньої реалізації та надання доступу через публічний інтерфейс.

*   **Поле `_data` в `Matrix`**: Масив даних матриці є приватним (`private readonly float[] _data`), доступ до елементів здійснюється через індексатор. Це дозволяє змінити внутрішнє представлення даних в майбутньому без впливу на зовнішній код.
    *   [DigitRecognitionNN/Models/Matrix.cs:7](DigitRecognitionNN/Models/Matrix.cs#L7) (Приватне поле)
    *   [DigitRecognitionNN/Models/Matrix.cs:11](DigitRecognitionNN/Models/Matrix.cs#L11) (Публічний індексатор)

*   **Ваги в `NeuralNetwork`**: Матриці ваг та зміщень є приватними полями. Зовнішній код взаємодіє з мережею через методи `Predict`, `TrainBatch` тощо.
    *   [DigitRecognitionNN/Models/NeuralNetwork.cs:8](DigitRecognitionNN/Models/NeuralNetwork.cs#L8)

## 4. DRY (Don't Repeat Yourself) - Не повторюй себе

Уникнення дублювання коду.

*   **Використання `TensorPrimitives`**: Замість написання циклів для базових операцій (додавання, множення, Softmax), використовуються методи з бібліотеки `System.Numerics.Tensors`, що зменшує дублювання логіки та покращує продуктивність.
    *   [DigitRecognitionNN/Models/Matrix.cs:101](DigitRecognitionNN/Models/Matrix.cs#L101) (`TensorPrimitives.Add`)
    *   [DigitRecognitionNN/Utils/ActivationFunction.cs:13](DigitRecognitionNN/Utils/ActivationFunction.cs#L13) (`TensorPrimitives.SoftMax`)

## 5. KISS (Keep It Simple, Stupid) - Роби це простіше

Код має бути максимально простим та зрозумілим.

*   **Реалізація `ReLu`**: Функція активації ReLu реалізована максимально просто та ефективно.
    *   [DigitRecognitionNN/Utils/ActivationFunction.cs:18](DigitRecognitionNN/Utils/ActivationFunction.cs#L18)

*   **Метод `Predict`**: Логіка прямого поширення (forward pass) записана послідовно та читабельно, відображаючи математичну модель.
    *   [DigitRecognitionNN/Models/NeuralNetwork.cs:39](DigitRecognitionNN/Models/NeuralNetwork.cs#L39)

## 6. Separation of Concerns - Розділення відповідальності

Різні аспекти функціональності програми повинні бути розділені.

*   **Розділення Моделі та Утиліт**: Логіка нейромережі (`NeuralNetwork`) відокремлена від математичних абстракцій (`Matrix`) та допоміжних функцій (`MathUtils`, `ActivationFunctions`).
    *   [DigitRecognitionNN/Models/NeuralNetwork.cs](DigitRecognitionNN/Models/NeuralNetwork.cs)
    *   [DigitRecognitionNN/Utils/ActivationFunction.cs](DigitRecognitionNN/Utils/ActivationFunction.cs)

## 7. Performance Optimization (Modern C# Practices)

Використання сучасних можливостей мови для покращення швидкодії.

*   **`Span<T>` та `TensorPrimitives`**: Використання `Span<float>` дозволяє працювати з пам'яттю без зайвих алокацій, а `TensorPrimitives` забезпечує SIMD-оптимізацію.
    *   [DigitRecognitionNN/Models/Matrix.cs:32](DigitRecognitionNN/Models/Matrix.cs#L32) (`AsSpan`)
    *   [DigitRecognitionNN/Models/Matrix.cs:137](DigitRecognitionNN/Models/Matrix.cs#L137) (Векторизоване множення матриць)

*   **`Parallel.For`**: Використання паралелізму для важких операцій, таких як множення матриць.
    *   [DigitRecognitionNN/Models/Matrix.cs:132](DigitRecognitionNN/Models/Matrix.cs#L132)
