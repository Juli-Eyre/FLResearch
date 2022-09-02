# FLResearch
Some paper about FL(attack, defense, HE)

[Conferences and Journals Collection for Federated Learning from 2019 to 2021 **含incentive works， Poisoning ，Personalization ，Optimization_Distribution， Communication](https://github.com/GillHuang-Xtler/flPapers)

* 除了poisoning works了解一些，incentive works，Optimization_Distribution蛮想了解和研究的

Advances and Open Problems in Federated Learning：==先做Privacy和Robust方面的，学习**Effificiency and Effectiveness**，**Ensuring Fairness and Addressing Sources of Bias**，**Addressing System Challenges **==
## FL(DNN)攻防
[Threats to Federated Learning: A Survey](https://arxiv.org/abs/2003.02133)

1. FL的数据重构攻击/梯度泄露攻击

   * [NIPS 2019 Deep Leakage from Gradients](https://proceedings.neurips.cc/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html) 通过加大训练的 batchsize 可以规避那些攻击。条件过于苛刻，例如要求恢复的数据样本数量要远小于总类别数目。

   * [NIPS 2020 Inverting Gradients - How easy is it to break privacy in federated learning?](https://proceedings.neurips.cc/paper/2020/hash/c4ede56bbd98819ae6112b20ac6bf145-Abstract.html) 在几个迭代或几个图像上平均梯度也不能保护用户的隐私，证明任何输入到全连接层的输入都可以被分析重建，与架构无关。

   * [CAFE: Catastrophic Data Leakage in Vertical Federated Learning NIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/08040837089cdf46631a10aca5258e16-Abstract.html) 利用 VFL 框架下 data index alignment，通过逐层的还原，实现了 VFL 阶段过程中 大批量训练数据 的还原。

   * [USENIX 2022 Exploring the Security Boundary of Data Reconstruction via Neuron Exclusivity Analysis](https://www.usenix.org/conference/usenixsecurity22/presentation/pan-exploring) 基于基本的梯度泄露攻击~~（原理太难看了..）~~

   * [Beyond Inferring Class Representatives: User-Level Privacy Leakage From Federated Learning 2019](https://ieeexplore.ieee.org/abstract/document/8737416)利用多任务GAN 训练，恢复 client 级别的多种隐私，包括 数据所属的类别、真假、来自哪一个 client，最后还能实现 某位client 训练数据的恢复。 缺陷是要求batchsize=1时才能恢复。

   * [A Novel Attribute Reconstruction Attack in Federated Learning 2021 ZJU](https://arxiv.org/abs/2108.06910) Attribute Reconstruction Attack 利用了受害者模型每轮更新的梯度，构建了一批虚拟样本并计算其对应的更新梯度，通过梯度上升优化得到 虚拟样本梯度和受害者模型上传的真实梯度的 cos 相似度的最大值：

     ![在这里插入图片描述](https://img.inotgo.com/imagesLocal/202203/27/202203270522586571_1.jpg)

     然后再更新

   * [Source Inference Attacks in Federated Learning](https://ieeexplore.ieee.org/abstract/document/9679121) 确定某一数据来源于哪一参与方，成员推理攻击不能

   * [GAN Enhanced Membership Inference: A Passive Local Attack in Federated Learning](https://ieeexplore.ieee.org/abstract/document/9148790 ICC 2020)   在FL场景下，各 client 所持有的数据存在分布不一致的情况，而之前的 shadow model 的方法要求 攻击者拥有部分和 target model 训练集同分布的辅助数据集，才能进行攻击；所以之前的方法在 FL 场景下的效果会大打折扣，或者说需要辅助信息（和 target model 训练集同分布的数据集）。本文介绍了一种利用 GAN 获取 整体参与者们 训练数据的分布，从而进行了更加精准的成员推断攻击
     [Efficient passive membership inference attack in federated learning](https://arxiv.org/abs/2111.00430) 相较于之前基于更新量的成员推断攻击，本文提出一种黑盒的利用 连续次更新量（结合了 time series） 的 推断攻击，仅需 **极小的计算量**（虽然攻击准确率差不多）。

2. Privacy:

   [Analyzing User-Level Privacy Attack Against Federated Learning CCF-A](https://ieeexplore.ieee.org/abstract/document/9109557) 首次尝试通过恶意服务器的攻击来探索用户级隐私泄露

   [A Framework for Evaluating Client Privacy Leakages in Federated Learning](https://link.springer.com/chapter/10.1007/978-3-030-58951-6_27) 分析本地训练的共享参数更新（来重建私有的本地训练数据。不同的超参数配置和攻击算法的不同设置对攻击有效性和攻击成本的影响。还评估了在使用通信效率高的FL协议时，不同梯度压缩率下的客户端隐私泄露攻击的有效性。

   [PMLR 2021 Gradient Disaggregation: Breaking Privacy in Federated Learning by Reconstructing the User Participant Matrix](https://proceedings.mlr.press/v139/lam21b.html) inference attack

   [Secure Aggregation is Insecure: Category Inference Attack on Federated Learning](https://ieeexplore.ieee.org/abstract/document/9618806 CCF-A2021) 类别隐私+Non-iid
   
   [usenix 2022 Label inference attacks against vertical federated learning](https://www.usenix.org/conference/usenixsecurity22/presentation/fu) vertical federated learning安全研究相对较少
   
3. 防御各类隐私攻击

   [Digestive neural networks: A novel defense strategy against inference attacks in federated learning COMPUTERS&Security2021 CCF-B](https://www.sciencedirect.com/science/article/pii/S0167404821002029) 
   本文在联邦学习场景下，提出了一种 Digestive neural networks （后称DNN，区别于传统的DNN），类似于输入数据的特征工程，用于“抽取原始数据的有效特征，并修改原始数据使之不同”，本地模型经过处理后的数据进行训练，从而大大降低了 FL 中各类推断攻击的成功率（假设server是攻击者）。

   [MixNN: Protection of Federated Learning Against Inference Attacks by Mixing Neural Network Layers2021](https://arxiv.org/abs/2109.12550)
   本文介绍了一种介于 client 和 server 中间的代理网络 MixNN proxy ，这样的代理网络有点类似同态加密（相当于利用神经网络进行加密），可以有效避免训练过程中的各类推断攻击。除此以外，本文还设计了一种利用 update 进行的 attribute 推断攻击，用于评估不同防御方法的防御效果。
   
    [Efficient and Privacy-Enhanced Federated Learning for Industrial Artificial Intelligence2019](https://ieeexplore.ieee.org/abstract/document/8859260) 本文提出了一种包含同态加密、差分隐私、多方安全计算的FL隐私保护框架。


4. 后门攻击：（和拜占庭攻击防御之间的关系？）

   https://github.com/THUYimingLi/backdoor-learning-resources 含Image and Video Classification attack-and-defense, Attack and Defense Towards Other Paradigms and Tasks(含FL), Evaluation, 是State-of-the-art一直在更新的

   * [usenix 2021 Blind Backdoors in Deep Learning Models](https://www.usenix.org/conference/usenixsecurity21/presentation/bagdasaryan) 和BadNets差不多，~~但是这是顶会？？~~

     Main task和后门效果差的太多了吧，这都能实现：

     ![image-20220730212735174](C:\Users\D2568\AppData\Roaming\Typora\typora-user-images\image-20220730212735174.png)

   * [AISTATS 20 How To Backdoor Federated Learning](https://proceedings.mlr.press/v108/bagdasaryan20a.html)

   * [dba distributed backdoor attacks against federated learning](https://github.com/GillHuang-Xtler/flPapers/blob/master/Poison/P15_dba_distributed_backdoor_attacks_against_federated_learning.pdf) ICLR 20 分布式后门

   * [联邦学习后门攻击总结（2019-2022）](https://blog.csdn.net/w18727139695/article/details/123727138)

   * [《Backdoor Learning: A Survey》阅读笔记](https://www.cnblogs.com/szhNJUPT/p/15552400.html)
     
     其他后门攻击：（感觉很AI了）
     
     [2019 ACM SIGSAC Latent Backdoor Attacks on Deep Neural Networks](https://dl.acm.org/doi/abs/10.1145/3319535.3354209) 结合迁移学习，~~感觉比顶会还难~~
      
   * Analyzing Federated Learning through an Adversarial Lens

    假设数据是iid的

    ![image-20220811170023327](C:\Users\D2568\AppData\Roaming\Typora\typora-user-images\image-20220811170023327.png)

    但是，这一方法会导致异常的梯度更新容易被基于准确度的方法检测出来。

    ![image-20220811170736205](C:\Users\D2568\AppData\Roaming\Typora\typora-user-images\image-20220811170736205.png)
    见上面github总结
     
     特定领域的后门攻击：
     
     [Backdoor Attack against Speaker Verification CCF-B](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2010.11607)
     
     [Backdoors in Federated Meta-Learning](https://arxiv.org/abs/2006.07026)

5. FL_Robust (model poisoning/后门攻击/拜占庭攻击)的防御

   * [联邦学习中的后门攻击的防御手段](https://blog.csdn.net/juli_eyre/article/details/124848370) 

   * [Robust Federated Training via Collaborative Machine Teaching using Trusted Instances](https://arxiv.org/abs/1905.02941) AAAI 20

   * 神经网络中后门攻击的防御：加噪声，梯度裁剪，阻碍模型将触发器标识为重要模块 Input Perturbation(如NeuralCleanse)，Model Anomalies(如SentiNet)，certify the computational graph, check during training..

   *  [Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning ](https://github.com/GillHuang-Xtler/flPapers/blob/master/Poison/P21-Manipulating-the-Byzantine-Optimizing-Model-Poisoning-Attacks-and-Defenses-for-Federated-Learning.pdf)NDSS 21


     what = "model poisoning + defense"
     goal = "untargeted under defense"
     why = "defenses exist"
     how = """
     poisoning: introduce perturbation vectors and optimize the scaling factor 
     defense: singular value decomposition based spectral methods
     
     This paper presents a generic framework for model poisoning attacks and a novel defense called divide-and-conquer (DnC) on FL. The key idea of its generic poisoning is that they introduce perturbation vectors and optimize the scaling factor $\gamma$ in both AGR-tailored and AGR-agnostic manners. DnC applies a singular value decomposition (SVD) based spectral methods to detect and remove outliers.

  * Local Model Poisoning Attacks to Byzantine-Robust Federated Learning

  虽然本文通篇都在讲local model，但本质上local model就是梯度数据，因此要改造的还是梯度数据，也就是说，**依然是在梯度数据的处理上做文章。**

  定义一个方向量，1表示当前梯度增加，-1表示当前梯度减小，其次定义攻击前的梯度与攻击后的梯度，那么优化问题的实质就是，**使攻击后的梯度与攻击前的梯度差别尽量大。**

  ![image-20220811163311397](C:\Users\D2568\AppData\Roaming\Typora\typora-user-images\image-20220811163311397.png)

  **Attacking Krum**

  由于Krum是选择一个最接近的local model作为返回结果，那么就尽量在攻击时让FL选择被compromise的设备的local model。这种情况下一共分为两步。首先，要让compromise设备的local model偏差尽量大，其次，要让其他compromise设别的local model都接近这个偏差值，这样方便FL依照Krum原则对他进行选择，从而达到攻击目的。

  Trimmed mean类似+数学

  防御方法：

  * Error Rate based Rejection (ERR)：对于每个本地模型，当包括本地模型时，使用聚合规则来计算全局模型A，而当排除本地模型时，再使用聚合规则来计算全局模型B，然后计算全局模型A和B在验证数据集上的错误率，抛弃掉对**错误率**有大幅影响的本地模型。
  
  * Loss Function based Rejection (LFR)：与ERR类似，只是在验证数据集上计算模型的交叉熵损失函数值，抛弃掉对**loss值**有大幅影响的本地模型。

   * Trojan Attack特洛伊攻击

     不依赖于对训练集的访问。相反，它们通过不使用任意触发来改进触发的生成，而是根据将导致内部神经元最大响应的值来设计触发。这在触发和内部神经元之间建立了更强的连接，并且能够以较少的训练样本注入有效的(＞98％)后门
     
 ### VFL&Inference attack
垂直联邦学习（Vertical Federated Learning，VFL）：各个参与者的数据集在样本空间一致，但在特征空间不同；例如某银行与某电商平台之间的合作——**两者的用户群体存在重叠，但是各自拥有不同的用户特征。**

![image-20220811145359945](C:\Users\D2568\AppData\Roaming\Typora\typora-user-images\image-20220811145359945.png)

##### Label Inference Attacks Against Vertical Federated Learning 2022 USENIX

* 现有的VFL架构对于标签的存储使用具有间接的隐私泄露风险：架构中由参与者维护的底层模型参数、训练算法中的梯度交换机制都有可能被潜在的恶意参与者利用，以窃取服务器端的标签数据。

  ![image-20220811141924490](C:\Users\D2568\AppData\Roaming\Typora\typora-user-images\image-20220811141924490.png)

  在图1所示的VFL体系结构中，只有一个参与者拥有标签，这与HFL不同，HFL中每个参与者都有自己的标签样本。确保私人标签的隐私是VFL提供的基本保证，因为标签可能是参与者的关键资产或高度敏感。此外，敌对参与者可能会试图利用被盗的标签建立类似的业务，或将敏感标签出售给地下行业。

  本文提出三种标签推断攻击：

  * 基于模型补全和半监督学习的被动攻击

  被动攻击：攻击方是Passive Party **(honest but curious)**，持有本地训练好的单个底部模型，**目的**是去推理自己训练集内的标签信息和其他参与方持有的标签信息。

  Method：在VFL中，attacker (Passive Party) 通过补全(Model Completion)本地的bottom模型（即给本地的bottom模型添加分类层），借助少量有标签的样本使用半监督学习算法（如MixMatch）对没有标签的样本进行预测，预测结果为伪标签，预测标签的精确程度通过不断fine-tune 补全后模型的分类层提高。最后，攻击方得到trained的完整模型(top + 其他参与方的bottom model)，输入本地feature到trained的完整模型，即可推断标签。

  * 基于本地恶意优化器的主动攻击

  主动攻击：攻击方是Passive Party **(malicious)**

  Method：在VFL中，attacker (Passive Party) 可以通过恶意修改优化器，加快bottom model梯度下降的速度，获得当前轮Top model参数优化时的优先权，使得attacker的本地bottom model能够**间接**获得“更倾向的“特征提取能力，以增强上面基于Model Completion补全的攻击方法的效果

  * 基于梯度推算的直接攻击

  受到相关工作《iDLG: Improved Deep Leakage from Gradients》的启发，本文进一步研究发现，对于不设置top model的VFL架构，可以直接依据服务器端反传的梯度的符号，推算出当前训练数据的标签。因为基于直接推算，此直接攻击方法在实验涉及的所有数据集上取得了100%的标签推断成功率。

  ![image-20220811153359131](C:\Users\D2568\AppData\Roaming\Typora\typora-user-images\image-20220811153359131.png)

  另外，本文评估了4种主流联邦学习隐私增强方法：梯度加噪，梯度压缩，PPDL（Privacy-preserving deep learning）和梯度离散化。实验结果显示现有的这4种防御无法有效抵御标签推断攻击：无法做到既基本维持模型在原任务上的性能，又有效降低标签推断攻击的推断成功率。

  ##### CAFE: Catastrophic Data Leakage in Vertical Federated Learning NeurIPS 2021

  ![image-20220811142544179](C:\Users\D2568\AppData\Roaming\Typora\typora-user-images\image-20220811142544179.png)

  ![image-20220811144052555](C:\Users\D2568\AppData\Roaming\Typora\typora-user-images\image-20220811144052555.png)

  在训练过程中，当地工人与他人交换他们的中间结果，以计算梯度并上传它们。因此，**服务器可以访问模型参数及其梯度。**由于数据垂直分区不同的工人，对于每个批，服务器（作为攻击者）需要发送 a data index or data id list到所有本地工人，以确保data with the same id sequence被每个工人selected.(我们将这一步命名为**数据索引对齐**)。数据索引对齐是垂直训练过程中不可避免的一个步骤，这为服务器（攻击者）提供了控制所选的批处理数据索引的机会。

  ![image-20220811151345218](C:\Users\D2568\AppData\Roaming\Typora\typora-user-images\image-20220811151345218.png)

  ![image-20220811152831590](C:\Users\D2568\AppData\Roaming\Typora\typora-user-images\image-20220811152831590.png)

  **逐层精确恢复：**

  ![image-20220811152937991](C:\Users\D2568\AppData\Roaming\Typora\typora-user-images\image-20220811152937991.png)

  我们还提出了一种有效的对策，利用假梯度(附录F)来降低CAFE的潜在风险。
  
  ##### Auditing Privacy Defenses in Federated Learning via Generative Gradient Leakage CVPR 2022
  

在这项工作中，我们验证了私人训练数据在特定的防御设置下仍然可以泄漏，并具有一种新的泄漏类型，即生成梯度泄漏(GGL)。与现有的仅依赖梯度信息来重建数据的方法不同，我们的方法利用了从公共图像数据集学习到的生成式对抗网络(GAN)的潜在空间，以补偿梯度退化过程中的信息损失。为了解决梯度算子和GAN模型引起的非线性问题，我们探索了各种无梯度优化方法（如进化策略和贝叶斯优化），并证明了它们与基于梯度的优化方法相比，在梯度重建高质量图像方面的优越性。我们希望该方法可以作为一个经验测量隐私泄漏量的工具，以促进设计更稳健的防御机制。

![image-20220811185923924](C:\Users\D2568\AppData\Roaming\Typora\typora-user-images\image-20220811185923924.png)

![image-20220811181234250](C:\Users\D2568\AppData\Roaming\Typora\typora-user-images\image-20220811181234250.png)

![image-20220811190044981](C:\Users\D2568\AppData\Roaming\Typora\typora-user-images\image-20220811190044981.png)

**Gradient Matching Loss**：目标函数（公式4）中的第一项鼓励求解器通过最小化生成图像的转换梯度和观察到的梯度˜之间的距离，在生成器的潜在空间中找到与客户的私人训练图像上下文相似的图像。

**Regularization Term**：仅使用梯度匹配损失进行优化很可能会产生偏离生成器的潜在分布的潜在向量，从而导致具有显著伪影的不现实图像。为了避免这个问题，我们在优化过程中探索了以下损失函数来正则化潜在向量

**Optimization Strategy.** 由于此优化问题非凸，选取合适的优化策略对于求解后生成的图像质量非常重要。此前梯度攻击中多选取基于梯度的优化算法，如 Adam 和 L-BFGS。然而这类优化器的效果非常依赖起始点的选择，往往需要多次尝试才能找到相对合适的解。并且我们发现，对于复杂的生成器，梯度优化算法非常容易收敛至局部最优，导致最后还原效果很差。因此，我们探索了两种无梯度的优化算法，即 Bayesian Optimization (BO) 和 Covariance Matrix Adaptation Evolution Strategy (CMA-ES)。

 ## 之前工作的总结

FL的抗投毒：相当于复现了该论文 [Privacy-Enhanced Federated Learning Against Poisoning Adversaries](https://ieeexplore.ieee.org/abstract/document/9524709)（CCF-A）

####  遗留问题

   * 抗投毒/抗后门攻击方面：当恶意work超过50%时，我们做的，以及Krum等防御方法不起作用。

     [usenix 2020 Justinian’s GAAvernor Robust Distributed Learning](https://www.usenix.org/conference/usenixsecurity20/presentation/pan) 结合了强化学习可以实现仅有一个良性work时也能抵御拜占庭攻击。

   * 效率方面：使用了同态加密（防止泄露隐私），耗时特别长

   * 假设了数据分布是IID的（联邦学习局部梯度成为全局梯度无偏估计的前提，使用Person相关系数检测出恶意梯度的前提）。猜很多防御方法都用了IID这个假设前提

   * 假设了work能投毒，其他服务器是诚实好奇的
