# Optimization tools

Это фреймворк для проведения оптимизационных расчетов. Состоит из классов оптимизаторов, а также абстрактных классов, которые необходимо переопределить в пользовательском коде для взаимодействия с оптимизаторами.

**Table of Contents**

- [Основная идея и возможности фреймворка](#основная-идея-и-возможности-фреймворка)
- [Используемые технологии](#используемые-технологии)
- [Установка](#установка)
- [Использование](#использование)

## Основная идея и возможности фреймворка
Оптимизационный алгоритм работает с абстрактными данными. Для того, чтобы решать на нем реальные задачи с возможностью гибкого выбора оптимизируемых параметров, солверов для проведения единичных расчетов и распараллеливания неободимо добавлять большое количество обвеса, который при изменении требований конкретной задачи приводит к большому переделыванию кода. В связи с этим возникла идея универсализовать способ проведения оптимизационных расчетов с учетом той вариативности, с которой столкнулся автор в своей работе. Вариативность заключается в:
- выборе оптимизатора под требуемую задачу и возможности комбинирования оптимизаторов
- возможности гибкого выбора параметров оптимизации и функций-ограничений в клиентском коде
- выборе решателя для единичных расчетов для задачи без внесения правок в код
- возможности проведения параллельных расчетов в рамках одной машины и кластера машин без внесения правок в код

## Используемые технологии
Градиентная оптимизация - scipy.minimize.slsqp
Удаленные расчеты - RabbitMQ

## Установка
В директорию проекта копируем папку с фреймворком и выполняем 
```bash
python -m venv venv
.\venv\Scripts\activate
python -m pip install -r ./optimization_tools/requirements.txt
```

## Использование
Самый простой пример использования - на функции Розенброка из доки по scipy.optimize

```bash
def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

```

создаем класс, отвечающий за оптимизируемый объект. Данный случай самый простой, наследуемся от AbstractObject

```bash
class SimpleVector(AbstractObject):
    def __init__(self, x1, x2) -> None:
        super().__init__()
        self.x1 = x1
        self.x2 = x2
```

Создаем класс решателя единичного расчета, отнаследованный от AbstractSolver, переопределяем метод solve. Названия функций-ограничений и целевой функции - mass должны соответствовать именам, которые мы далее определим в условиях оптимизации
```bash
class RosenSolver(AbstractSolver):
    def solve(self, calc_task: SimpleVector, unique_id: str, res_type: str | None):
        try:

            if res_type == "mass":
                return rosen(np.array([calc_task.x1, calc_task.x2]))
            elif res_type == "ineq1":
                return np.array([1 - calc_task.x1 - 2*calc_task.x2,
                                            1 - calc_task.x1**2 - calc_task.x2,
                                            1 - calc_task.x1**2 + calc_task.x2])[0]
            elif res_type == "ineq2":
                return np.array([1 - calc_task.x1 - 2*calc_task.x2,
                                            1 - calc_task.x1**2 - calc_task.x2,
                                            1 - calc_task.x1**2 + calc_task.x2])[1]
            elif res_type == "ineq3":
                return np.array([1 - calc_task.x1 - 2*calc_task.x2,
                                            1 - calc_task.x1**2 - calc_task.x2,
                                            1 - calc_task.x1**2 + calc_task.x2])[2]
            elif res_type == None:
                return {"ineq1": 1 - calc_task.x1 - 2*calc_task.x2,
                        "ineq2": 1 - calc_task.x1**2 - calc_task.x2,
                        "ineq3": 1 - calc_task.x1**2 + calc_task.x2,
                        "mass": rosen(np.array([calc_task.x1, calc_task.x2])) }
            else:
                raise NameError
        except Exception as exc:
            raise SolverError from exc
```

Создаем объект OptConditions, хранящий переменные оптимизации с их диапазоном зхначений и функции ограничения
```bash
x_obj = SimpleVector(0.4, 0.1)
opt_vars = {"x1": {"min": 0, "max": 1}, "x2": {"min": -0.5, "max": 2}}
constraints = {"ineq1": 0, "ineq2": 0, "ineq3": 0,}
opt_params = OptConditions(opt_vars, constraints)
```

Создаем объект солвера
```bash
rosen_solver_obj = RosenSolver()
```

Создаем таску и оптимизатор
```bash
optimization_task = OptimizationTaskWithNormalization(initial_state=x_obj,
         unique_id="1", opt_conditions=opt_params,
        solver=rosen_solver_obj
        )
optimizer = GradientOptimizer(optimization_task)
```

Запускаем оптимизацию и получаем результаты
```bash
result: OptimizationTaskResults = outer_optimizer.run_optimization()
```

В случае, если целевая функция плохо поддается градиентной оптимизации, можно запустить оптимизацию через BruteForceOptimizer. Класс таски здесь OptimizationTaskWithInnerOptimizer - по названию видно, что она позволяет добавить внутренний оптимизатор, однако в данном примере это не используется. Также задается дискретность, в данном случае диапазон каждой из переменных делится на 100 равномерных точек. 

```bash
optimization_task = OptimizationTaskWithInnerOptimizer(initial_state=x_obj,
         unique_id="1", opt_conditions=opt_params,
        solver=rosen_solver_obj, inner_optimizer=None
        )
optimizer = BruteForceOptimizer(optimization_task, discreteness=100)
```

Для задач комбинированного типа, в которых присутствуют как переменные, по которым функция является непрерывной, так и переменные, по которым функция разрывна, можно задать оптимизацию с вложенной структурой. 

Рассмотрим более сложные случаи настройки оптимизационного процесса 
```bash
```
```bash
```