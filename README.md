# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev] 
Отчет по лабораторной работе #3 выполнил(а):
- Ермолинский Семён Михайлович
- РИ000024 

Отметка о выполнении заданий :

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | # | 20 |



## Цель работы
Познакомиться с программными средствами для создания системы машинного обучения и ее интеграции в Unity.

## Задание 1
### Реализовать систему машинного обучения в связке Python - Google-Sheets – Unity.

1.1 В проект unity добавим ml-agents-release_19/com.unity.ml-agents/package.json и ml-agents-release_19/com.unity.ml-agents.extensions/package.json
![Снимок экрана 2022-10-19 в 16 48 14](https://user-images.githubusercontent.com/100123572/196971030-15a8b93e-aa40-4bb9-bfc8-da2e4e3b408e.png)

1.2 Создадим виртуальное окружение и скачаем в него mlagents 0.28.0 и torch 1.7.1
![Снимок экрана 2022-10-19 в 16 49 29](https://user-images.githubusercontent.com/100123572/196971206-98c8714a-dd06-4e71-934b-3f564d778015.png)
![Снимок экрана 2022-10-19 в 16 49 40](https://user-images.githubusercontent.com/100123572/196971210-01703952-1ed8-41cc-a870-99cd0ccdd12f.png)

1.3 Создание сцены, куба и шара
![Снимок экрана 2022-10-19 в 16 50 04](https://user-images.githubusercontent.com/100123572/196971297-bc68d1ac-bc3b-4910-a325-1921bc2cd4a4.png)

1.4 Добавим сфере скрипт RollerAgent.cs

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    Rigidbody rBody;
    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public Transform Target;
    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        Target.localPosition = new Vector3(Random.value * 8-4, 0.5f, Random.value * 8-4);
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target.localPosition);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }
    public float forceMultiplier = 10;
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

        if(distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }
}

1.6 В корень проекта добавим файл конфигурации нейронной сети и запустим работу ml-агена
![Снимок экрана 2022-10-19 в 17 13 48](https://user-images.githubusercontent.com/100123572/196972020-23ddc9ba-f701-429d-a04c-6b51d8afe357.png)
![Запись экрана 2022-10-19 в 17 08 48](https://user-images.githubusercontent.com/100123572/196972161-6e500751-55d1-4561-a5e0-d2fd2efb3a0d.gif)

1.7 Сделаем несколько копий модели TargetArea, и обучим их
![Запись экрана 2022-10-19 в 17 18 43](https://user-images.githubusercontent.com/100123572/196972145-e58b1037-cb70-4d01-8792-5f4b158e4a86.gif)

1.8 Проверим работу полученной модели
![Запись экрана 2022-10-19 в 20 32 42](https://user-images.githubusercontent.com/100123572/196972282-0229f09c-711a-4cee-a63d-0c186d3f86a1.gif)

При увеличении количества копий, модель обучается быстрее.


## Задание 2
### Необходимо связать данные из ленейной регрессии с кодом и вывести loss в таблицу

Подробно описать каждую строку файла конфигурации нейронной сети. Самостоятельно найти информацию о компонентах Decision Requester, Behavior Parameters, добавленных сфере.

behaviors:
  RollerBall: # указываем id агента
    trainer_type: ppo # режим обучения (Proximal Policy Optimization)
    hyperparameters:
      batch_size: 10 # количество опытов на каждой итерации
      buffer_size: 100 # количество опыта, которое нужно набрать перед обновлением модели
      learning_rate: 3.0e-4 # начальная скорость обучения
      beta: 5.0e-4 # сила регуляции энтропии, увеличивает случайность действий
      epsilon: 0.2 # порог расхождений между старой и новой политиками при обновлении
      lambd: 0.99 # параметр регуляции, насколько сильно агент полагается на текущий value estimate
      num_epoch: 3 # количество проходов через буфер опыта, при выполнении оптимизации
      learning_rate_schedule: linear # определяет как скорость обучения изменяется с течением времени
                                     # linear линейно уменьшает скорость
    network_settings:
      normalize: false # отключаем нормализацию входных данных
      hidden_units: 128 # количество нейронов в скрытых слоях сети
      num_layers: 2 # количество скрытых слоёв в сети
    reward_signals:
      extrinsic:
        gamma: 0.99 # коэффициент скидки для будущих вознаграждений
        strength: 1.0 # коэффициент на который умножается вознаграждение
    max_steps: 500000 # общее количество шагов, которые должны быть выполнены в среде до завершения обучения
    time_horizon: 64 # сколько опыта нужно собрать для каждого агента, прежде чем добавлять его в буфер
    summary_freq: 10000 # количество опыта, который необходимо собрать перед созданием и отображением статистики






## Выводы

Я научился работать с ML агентом, понял как получить результат обучения. Что значат данные в yaml. Понял, что чем больше моделей задействованы в обучении, тем лучше будет результат обучения. Разобрался как создать пространство под агента, и как пользоваться консолью в anaconda.

Игровой баланс — в случае с компьютерными развлечениями, это равновесие некоторых характеристик, механик, персонажей, тактик, команд, правил, и то что связано непосредственно с процессом. Он существует в том или ином виде даже в одиночных играх, но по большей части это относится к многопользовательскому опыту.

Математика является одним из самых важных аспектов в создании баланса. В разных аспектах игры нужно получить такие параметры, чтобы попасть в "золотую середину" еще это называют состоянием потока. Это когда игрок полностью вовлечен в процесс, и он не скучает, а так же не стрессует. У этого процеса есть множество взаимосвязанных параметров и нет чёткого алгоритма получения качественного результата, поэтому при создании баланса сейчас часто применяют машинное обучение.
