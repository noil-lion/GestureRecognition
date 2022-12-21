# Rehab-Net论文精读小记
|name|value|说明|
|:--|:--|:--|
|Titile|Rehab-Net: Deep Learning Framework for Arm Movement Classification Using Wearable Sensors for Stroke Rehabilitation|Rehab-Net 一种用于分类基于可穿戴sensor的手臂运动的深度学习框架|
|Abstract|1. “Rehab-Net” could provide a measure of rehabilitation progress.<br>2.accurately monitor the clinical progress of the rehabilitated subjects under the ambulatory settings.|关键句摘录|
|研究问题|1. 中风后康复训练能提升ADL能力，但临床康复耗时、贵、不便，home/remote-based 康复技术可监测康复执行和康复进程。<br> 2. HAR可以作为ADL实时监测的solution，这其中穿戴惯性sensor应用最广，其面临的挑战是构建一个在实时场景下的精确识别分类器||
|解决方案|1.提出Rehab-Net康复框架，构建有效的、个性化、低复杂度端到端分类模型。2.分别从数据集<半自然和全自然不可控环境>、数据预处理优化、模型结构优化、超参数优化、训练框架等对定制化CNN模型进行优化以达到上述目标。||
|方法论细节|1. 网络结构：Conv（20，9，1）- Relu - Pool（2，1）-Conv（20，9，1）- Relu - Pool（2，1）-FC（n）;<br> 2.预处理优化：四个样本求平均，下采样同时保留有用信息；三轴加速度数据模融合为1D信号数据，减小网络计算量 3. 个性化训练：基于个人数据进行训练和参数调整，避免过拟合，使用dropout和数据增强（添加随机噪声）||
|创新点|||
|结论|1.可借鉴文中的问题引入：如临床康复存在的限制，深度框架的价值，其更具泛化性和可适应（不同对象差异如震颤程度、康复阶段、个体差异）。2. 方法论的提出和优化方向，如个性化参数训练框架、计算资源优化提升可集成性（计算复杂度计算方式可借鉴）、网络结构参数和训练参数的优化方向；||
|待解决问题|更全面的与传统方案的对比，和更丰富的对象和数据||

## 关键句
1. The principal objective of activity recognition is to identify human behavior in real time scenario to provide proactive assistance to users even outside the conventional clinical setting。  
   real time scenario 实时场景  
   proactive assistance 主动积极的协助  
   译：活动识别的主要目标是实时场景中识别人类行为，以便在传统临床环境之外为用户提供积极的帮助

2. diagnosis and prognosis reliability 诊断和预后可靠性
3. hence have been the preferred modality in HAR applications 因此一直是HAR应用的首选模式
4. 