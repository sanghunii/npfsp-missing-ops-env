## Schematic

아래는 본 연구의 NPFSP 환경의 schematic이다.
![NPFSP Environment Schematic](./assets/env_schematic.png)


## Assumptions

본 연구에서 다루는 Scheduling Environment는 **NPFSP(Non-Permutation Flowshop Scheduling Problem)** 의 일반적인 가정¹²과 더불어, Limited Buffers 및 Window-based Inspection 제약에 따른 몇 가지 추가적인 가정을 전제로 한다.

> ¹ *Multi-heuristic desirability ant colony system heuristic for non-permutation flowshop scheduling problems (The International Journal of Advanced Manufacturing Technology, 2007)*
> ² *Solving non-permutation flow-shop scheduling problem via a novel deep reinforcement learning approach (Computers & Operations Research, 2023)*

- **무한한 초기 버퍼:** 모든 작업은 시작 시간(Start time)에 첫 번째 기계의 버퍼에서 작업 가능한 상태로 대기한다. 이때 첫 번째 기계의 대기 버퍼는 무한한 용량을 갖는다고 가정한다.
- **단일 처리 원칙:** 각 기계와 검사 기계는 한 번에 단 하나의 작업만 처리할 수 있으며, 각 작업 또한 한 번에 하나의 기계에서만 처리될 수 있다.
- **비선점형(Non-preemptive) 작업:** 모든 작업은 한 번 기계 또는 검사 기계에서 처리가 시작되면 완료될 때까지 중단되거나 분할될 수 없다.
- **Sequence-independent 셋업:** 기계의 셋업 시간 및 작업의 이동 시간은 가공 시간 $p_{ij}$ 및 검사 시간 $I_{ik}$에 이미 포함된 것으로 간주하며, 이는 작업 순서에 영향을 받지 않는다.
- **제한된 버퍼 및 Blocking:** 첫 번째 기계를 제외한 모든 후속 기계 $j \in \{2, 3, \dots, m\}$ 및 검사 기계 $k \in \mathcal{I}$ 앞에는 용량이 $C$로 제한된 대기 버퍼가 존재한다. 후속 기계의 버퍼가 가득 찬 경우, 선행 기계는 작업을 마쳤음에도 다음 단계로 배출하지 못하고 **Blocking** 상태가 된다.
- **Window 제약 기반 Missing Operation:** 본 스케줄링 환경에서의 결측 공정(Missing operation)은 Inspection Machine에 한하여 발생한다. 검사 공정은 Window 제약 $W$를 위반하지 않는 선에서 제한적으로 생략(Bypass)될 수 있으며, 검사가 생략된 작업은 선행 기계에서 완료된 즉시 다음 기계의 버퍼로 직접 라우팅된다.
- **동시 완료 시 라우팅 우선순위:** 특정 Stage에서 일반 기계(Main machine)와 검사 기계(Inspection machine)의 작업이 동시에 완료된 경우, 검사 기계에서 완료된 작업이 다음 기계로 넘어가는 라우팅 우선순위(Routing priority)를 갖는다.

---

## ⚙️ Core Environment Logic: `step()`

환경의 상태 전이(State Transition)를 담당하는 `step()` 함수는 Event-Driven 방식으로 설계되었으며, 크게 **4가지 Phase**로 나뉘어 실행된다. 

---

### Phase 1. Action Processing & Buffer Sorting
Agent로부터 전달받은 Action을 해석하여 각 Machine의 Buffer를 정렬한다.
- **Buffer Sorting:** 전달받은 action 값을 기반으로 각 Machine 앞의 Buffer 대기열을 재정렬한다.
- **Action Masking (`-1`):** 현재 단계에서 작업 할당이 필요 없는(작업 중이거나 Blocked 상태인) Machine은 `-1`을 action으로 받는다. 즉, 직전 단계에서 `available_machines[m] == True`였던 Machine들만 유효한 action을 수행한다. <br>
이때 `available_machines[m] == False`인 machine이 -1 action을 받지 않아도 상관 없으며 이후 이어질 로직에서 해당 machine에 대한 실질적인 job 투입은 일어나지 않는다.

### Phase 2. Job Allocation (Reverse Order)
정렬된 Buffer를 바탕으로 Machine에 Job을 투입한다. 병목 현상(Blocking)을 정확히 모사하기 위해 **마지막 Machine부터 역순(Backward)** 으로 할당을 진행한다.

1. **Buffer 공간 확보:** $m$번째 Machine에 Job이 할당되면 해당 Machine의 Buffer에 빈자리가 1개 발생한다.
2. **우선순위 기반 Job 이동:** 빈자리가 생겼을 때, 직전 단계에서 잔여 작업 시간(`remain_time`)이 `0`임에도 Blocked 상태였던 Job들을 끌어온다.
   - **Priority 1:** 직전 Inspection Machine에 대기 중인 Job (`remain_time == 0`)
   - **Priority 2:** $m-1$번째 Main Machine에 대기 중인 Job (`remain_time == 0`)
   - 위 조건에 해당하면 Job을 $m$번째 Buffer로 이동시킨 후, $m-1$번째 Machine에 새로운 Job 투입을 진행한다.

### Phase 3. Time Progression & Event Evaluation
다음 의사결정 시점(`transition_point == True`)이 도달할 때까지 가상 시간을 흐르게 하며 Job을 물리적으로 이동시킨다. 
*(조건: `available_machines` 중 하나라도 `True`가 될 때까지 반복)*

#### 3-1. Time Update
- **Processing Time 계산:** 현재 작업 중인 Job들의 잔여 시간 중 **최솟값(Min)**만큼 시간을 진행시킨다.
- **Metrics 누적:** 진행된 시간만큼 각 Machine의 Idle time과 Blocked time을 누적한다. (Agent의 보상/상태 정보로 활용)
- **Remain Time 갱신:** 각 Machine의 잔여 작업 시간을 차감한다.

#### 3-2. Physical Job Movement (Reverse Order)
시간 갱신 후 `remain_time == 0`이 된 Job들을 다음 단계로 이동시킨다. 이 역시 마지막 Machine부터 역순으로 검사한다.

**A. Last Machine**
- Agent의 의사결정이 개입하지 않는다.
- 작업이 끝난 Job이 전체의 마지막 Job이 아니면, Buffer 상태에 따라 대기 중인 Job을 즉시 투입하거나 대기(Idle) 상태로 전환한다.

**B. Inspection Machine ($m$)**
- **$m+1$ Machine Buffer Empty:** Job을 $m+1$로 투입한다. (이때 $m+1$이 Last Machine이고 비어있다면 즉시 작업 시작)
- **$m+1$ Machine Buffer Full:** Job은 이동하지 못하고 **Blocked** 처리된다.
- 투입 성공 시, 빈자리에 Inspection 대기 중인 Job을 바로 할당하며, $m+1$ Buffer가 비어있었다면 연쇄 작용을 위해 `trigger_wakeup = True`를 발생시킨다.

> 💡 **[Rule] Inspection Priority:** 
> Inspection Machine은 Main Machine보다 Job을 다음 단계로 넘기는 데 **우선권**을 갖는다. $m+1$ Buffer에 1자리만 남았고 $m$과 Inspection Machine 모두 `remain_time == 0`이 되었다면, Inspection Machine의 Job이 먼저 이동하고 $m$ Machine은 Blocked 된다.

**C. Main Machine ($m$)**
- **Inspection 대상 Job (Forced/Bypass):** Inspection Buffer에 자리가 있으면 투입, 비어있다면 즉시 할당한다.
- **Inspection 비대상 Job:** $m+1$ Machine Buffer 상태에 따라 투입하거나 Blocked 처리된다. ($m+1$ Buffer가 비어있었다면 `trigger_wakeup = True`)

#### 3-3. Transition Flag Update (`available_machines`)
- **Wake-up Check (`trigger_wakeup`):** $m+1$ Machine에 Job이 투입되어 상태가 변했을 때, 이전 Machine들의 Blocking 상태가 해제될 수 있는지 미래 상태(`b_m2_will_open` 등)를 확인하여 `available_machines` 플래그를 `True`로 갱신한다.
- **Current Machine Check:** $m$ Machine이 비어있고 Buffer에 대기 Job이 있거나, Blocked 상태지만 다음 루프에서 자리가 날 예정(연쇄 해제)이라면 `available_machines[m] = True`로 설정한다.

### Phase 4. State Generation
`transition_point`가 활성화되어 루프를 빠져나오면, 업데이트된 환경을 바탕으로 State Value들을 계산한다.
- 각 Machine의 Buffer 상태, 누적 Idle/Blocked 시간, Job의 잔여 시간 등을 종합하여 **State Vector**를 생성하고 Agent에게 반환(`return`)한다.
