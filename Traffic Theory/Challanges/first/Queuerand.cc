#include <omnetpp.h>

using namespace omnetpp;

class Queuerand : public cSimpleModule
{
protected:
    cMessage *msgInServer; // 当前正在服务的消息
    cMessage *endOfServiceMsg; // 服务结束的消息

    cQueue Queue; // 队列，用于存储消息

    simsignal_t qlenSignal; // 队列长度信号
    simsignal_t busySignal; // 服务器忙碌状态信号
    simsignal_t queueingTimeSignal; // 排队时间信号
    simsignal_t responseTimeSignal; // 响应时间信号

    double avgServiceTimer; // 平均服务时间

    bool serverBusy; // 服务器是否忙碌

public:
    Queuerand();
    virtual ~Queuerand();

protected:
    virtual void initialize() override; // 初始化模块
    virtual void handleMessage(cMessage *msg) override; // 处理消息
    void startPacketService(cMessage *msg); // 开始处理消息
    void putPacketInQueue(cMessage *msg); // 将消息放入队列
    cMessage* selectRandomPacket(); // 随机选择一个消息
};

Define_Module(Queuerand);

Queuerand::Queuerand()
{
    msgInServer = endOfServiceMsg = nullptr;
}

Queuerand::~Queuerand()
{
    delete msgInServer;
    cancelAndDelete(endOfServiceMsg);
}

void Queuerand::initialize()
{
    endOfServiceMsg = new cMessage("end-service");
    Queue.setName("Queuerand");
    serverBusy = false;

    // 注册信号
    qlenSignal = registerSignal("qlenr");
    busySignal = registerSignal("busyr");
    queueingTimeSignal = registerSignal("queueingTimer");
    responseTimeSignal = registerSignal("responseTimer");

    // 初始化信号值
    emit(qlenSignal, Queue.getLength());
    emit(busySignal, serverBusy);

    // 获取平均服务时间参数
    avgServiceTimer = par("avgServiceTimer").doubleValue();
}

void Queuerand::handleMessage(cMessage *msg)
{
    if (msg->isSelfMessage()) { // 当前服务器中的消息服务完成

        // 记录完成服务
        EV << "Completed service of " << msgInServer->getName() << endl;

        // 发送处理完成的消息到下游模块
        send(msgInServer, "out");

        // 发射响应时间信号
        emit(responseTimeSignal, simTime() - msgInServer->getTimestamp());

        // 如果队列不为空，开始处理下一个消息
        if (!Queue.isEmpty()) {
            // 随机选择一个消息
            msgInServer = selectRandomPacket();

            // 发射队列长度和排队时间信号
            emit(qlenSignal, Queue.getLength());
            emit(queueingTimeSignal, simTime() - msgInServer->getTimestamp());

            // 开始处理消息
            startPacketService(msg);
        } else {
            // 队列为空，服务器变为空闲状态
            msgInServer = nullptr;
            serverBusy = false;
            emit(busySignal, serverBusy);

            // 记录服务器空闲
            EV << "Empty Queue, server goes IDLE" << endl;
        }

    } else { // 从源到达的消息

        // 设置到达时间戳
        msg->setTimestamp();

        if (serverBusy) {
            // 如果服务器忙碌，将消息放入队列
            putPacketInQueue(msg);
        } else {
            // 如果服务器空闲，直接开始服务
            msgInServer = msg;
            startPacketService(msg);

            // 服务器状态变为忙碌
            serverBusy = true;
            emit(busySignal, serverBusy);

            // 排队时间为零
            emit(queueingTimeSignal, SIMTIME_ZERO);
        }
    }
}

void Queuerand::startPacketService(cMessage *msg)
{
    // 生成服务时间并安排服务完成事件
    simtime_t serviceTime = exponential(avgServiceTimer);
    scheduleAt(simTime() + serviceTime, endOfServiceMsg);

    // 记录开始服务
    EV << "Starting service of " << msgInServer->getName() << endl;
}

void Queuerand::putPacketInQueue(cMessage *msg)
{
    Queue.insert(msg); // 将消息插入队列
    emit(qlenSignal, Queue.getLength()); // 发射队列长度信号

    // 记录消息进入队列
    EV << msg->getName() << " enters Queuerand" << endl;
}

cMessage* Queuerand::selectRandomPacket()
{
    // 从队列中随机选择一个消息
    int randomIndex = intuniform(0, Queue.getLength() - 1); // 随机索引
    cMessage *selectedMsg = (cMessage *)Queue.remove(Queue.get(randomIndex));

    // 记录随机选择
    EV << "Randomly selected " << selectedMsg->getName() << " from Queue" << endl;

    return selectedMsg;
}
