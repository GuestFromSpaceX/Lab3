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

1 В проект unity добавим ml-agents-release_19/com.unity.ml-agents/package.json и ml-agents-release_19/com.unity.ml-agents.extensions/package.json
![Снимок экрана 2022-10-19 в 16 48 14](https://user-images.githubusercontent.com/100123572/196971030-15a8b93e-aa40-4bb9-bfc8-da2e4e3b408e.png)

2 Создадим виртуальное окружение и скачаем в него mlagents 0.28.0 и torch 1.7.1
![Снимок экрана 2022-10-19 в 16 49 29](https://user-images.githubusercontent.com/100123572/196971206-98c8714a-dd06-4e71-934b-3f564d778015.png)
![Снимок экрана 2022-10-19 в 16 49 40](https://user-images.githubusercontent.com/100123572/196971210-01703952-1ed8-41cc-a870-99cd0ccdd12f.png)

3 Создание сцены, куба и шара
![Снимок экрана 2022-10-19 в 16 50 04](https://user-images.githubusercontent.com/100123572/196971297-bc68d1ac-bc3b-4910-a325-1921bc2cd4a4.png)

4 Добавим сфере скрипт RollerAgent.cs

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

5 В корень проекта добавим файл конфигурации нейронной сети и запустим работу ml-агена
![Снимок экрана 2022-10-19 в 17 13 48](https://user-images.githubusercontent.com/100123572/196972020-23ddc9ba-f701-429d-a04c-6b51d8afe357.png)
![Запись экрана 2022-10-19 в 17 08 48](https://user-images.githubusercontent.com/100123572/196972161-6e500751-55d1-4561-a5e0-d2fd2efb3a0d.gif)

6 Сделаем несколько копий модели TargetArea, и обучим их
![Запись экрана 2022-10-19 в 17 18 43](https://user-images.githubusercontent.com/100123572/196972145-e58b1037-cb70-4d01-8792-5f4b158e4a86.gif)

7 Проверим работу полученной модели
![Запись экрана 2022-10-19 в 20 32 42](https://user-images.githubusercontent.com/100123572/196972282-0229f09c-711a-4cee-a63d-0c186d3f86a1.gif)

При увеличении количества копий, модель обучается быстрее.


## Задание 2
### Подробно описать каждую строку файла конфигурации нейронной сети. Самостоятельно найти информацию о компонентах Decision Requester, Behavior Parameters, добавленных сфере.

![Снимок экрана 2022-10-20 в 17 22 21](https://user-images.githubusercontent.com/100123572/196975274-06ea572a-97ff-4dc0-881c-73ace3fd35ef.png)





## Выводы

Я научился работать с ML агентом. Что значат данные в yaml. Понял, что чем больше моделей задействованы в обучении, тем лучше будет результат обучения. Разобрался как создать пространство под агента, и как пользоваться консолью в anaconda.
