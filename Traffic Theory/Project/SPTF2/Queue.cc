#include <omnetpp.h>
#include <vector>
#include <random>

using namespace omnetpp;
/* I use a tree to build the queue */
// Define the node structure of a binary sorting tree
struct TreeNode {
    cMessage *msg; // Message pointer
    double servicetime; // Service time
    TreeNode *left;
    TreeNode *right;

    // Constructor function
    TreeNode(cMessage *m, double st) : msg(m), servicetime(st), left(nullptr), right(nullptr) {}
};

// Define binary sorting tree class
class BinarySearchTree
{
private:
    TreeNode *root; // Root of the tree
    cMessage *minMsg; // Min service time -> message
    double minTime; // Min service time

    // Auxiliary function: Recursive insertion of nodes
    TreeNode *insertNode(TreeNode *node, cMessage *msg, double servicetime) {
        if (node == nullptr) {
            return new TreeNode(msg, servicetime);
        }
        if (servicetime < node->servicetime) {
            node->left = insertNode(node->left, msg, servicetime);
        } else {
            node->right = insertNode(node->right, msg, servicetime);
        }
        return node;
    }

    // Auxiliary function: Find the parent node of the minimum value node
    TreeNode *findMinNode(TreeNode *node, TreeNode *&parent) {
        while (node->left != nullptr) {
            parent = node;
            node = node->left;
        }
        return node;
    }

    // Auxiliary function: recursively count the number of nodes
    int countNodes(TreeNode *node) {
        if (node == nullptr) {
            return 0;
        }
        return 1 + countNodes(node->left) + countNodes(node->right);
    }

public:
    // Constructor function
    BinarySearchTree() : root(nullptr) {}

    // Insert Node
    void insert(cMessage *msg, double servicetime) {
        root = insertNode(root, msg, servicetime);
    }

    // Retrieve and delete the minimum service time node
    cMessage *extractMin() {
        if (root == nullptr) {
            return nullptr;
        }

        TreeNode *parent = nullptr;
        TreeNode *minNode = findMinNode(root, parent);

        // Retrieve the information of the smallest node
        cMessage *minMsg = minNode->msg;
        minTime = minNode->servicetime;

        // Delete the smallest node and rebuild the tree
        if (parent == nullptr) { // The minimum node is the root node
            root = minNode->right;
        } else {
            parent->left = minNode->right;
        }
        delete minNode;

        return minMsg;
    }

    // Extract the minimum service time
    double getMinSt(){
        return minTime;
    }

    // Count the number of nodes (excluding the root node)
    int countNodesExcludingRoot() {
        if (root == nullptr) {
            return 0;
        }
        return countNodes(root) - 1;
    }
};


class Queue : public cSimpleModule
{
protected:
    cMessage *msgInServer;
    cMessage *endOfServiceMsg;

    cQueue queue;

    simsignal_t qlenSignal;
    simsignal_t busySignal;
    simsignal_t queueingTimeSignal;
    simsignal_t responseTimeSignal;

    double avgServiceTime;
    bool serverBusy;
    BinarySearchTree bst;



public:
    Queue();
    virtual ~Queue();

protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
    void startPacketService(cMessage *msg);
    void putPacketInQueue(cMessage *msg);
};





Define_Module(Queue);


Queue::Queue()
{
    msgInServer = endOfServiceMsg = nullptr;
}

Queue::~Queue()
{
    delete msgInServer;
    cancelAndDelete(endOfServiceMsg);
}

void Queue::initialize()
{


    endOfServiceMsg = new cMessage("end-service");
    queue.setName("queue");
    serverBusy = false;

    //signal registering
    qlenSignal = registerSignal("qlen");
    busySignal = registerSignal("busy");
    queueingTimeSignal = registerSignal("queueingTime");
    responseTimeSignal = registerSignal("responseTime");

    //initial messages
    emit(qlenSignal, queue.getLength());
    emit(busySignal, serverBusy);

    //get avgServiceTime parameter
    avgServiceTime = par("avgServiceTime").doubleValue();

}

void Queue::handleMessage(cMessage *msg)
{
    if (msg->isSelfMessage()) { //Packet in server has been processed

        //log service completion
        EV << "Completed service of " << msgInServer->getName() << endl;

        //Send processed packet to sink
        send(msgInServer, "out");

        //emit response time signal
        emit(responseTimeSignal, simTime() - msgInServer->getTimestamp());
        EV << "quene nb:" << bst.countNodesExcludingRoot() << endl;
        //start next packet processing if queue not empty
        //if (!queue.isEmpty()) {
        if (bst.countNodesExcludingRoot()) { //if countNodesExcludingRoot = 0, means queue is empty
            //this is a queue
            //msgInServer = (cMessage *)queue.back();
            //queue.remove(queue.back());

            //Emit queue len and queuing time for this packet
            //emit(qlenSignal, queue.getLength());


            //start service
            //bst.insert(msg, exponential(avgServiceTime)); // add to tree
            startPacketService(msg);
            emit(qlenSignal, bst.countNodesExcludingRoot());
            emit(queueingTimeSignal, simTime() - msgInServer->getTimestamp());
            serverBusy=true;
            emit(busySignal, serverBusy);


        } else {
            //server is not busy anymore
            msgInServer = nullptr;
            serverBusy = false;
            emit(busySignal, serverBusy);

            //log idle server
            EV << "Empty queue, server goes IDLE" <<endl;
        }

    }
    else { //packet from source has arrived

        //Setting arrival timestamp as msg field
        msg->setTimestamp();

        if (serverBusy) {
            putPacketInQueue(msg);
        }
        else { //server idle, start service right away
            //Put the message in server and start service
            //msgInServer = msg;
            bst.insert(msg, exponential(avgServiceTime)); // add to tree
            startPacketService(msg);

            //server is now busy
            serverBusy=true;
            emit(busySignal, serverBusy);

            //queueing time was ZERO
            emit(queueingTimeSignal, SIMTIME_ZERO);
        }
    }
}

void Queue::startPacketService(cMessage *msg)
{

    //generate service time and schedule completion accordingly
    msgInServer = bst.extractMin();
    //simtime_t serviceTime = exponential(avgServiceTime);
    simtime_t serviceTime = bst.getMinSt();
    EV <<"service time" << serviceTime<< endl;
    scheduleAt(simTime()+serviceTime, endOfServiceMsg);

    //log service start
    EV << "Starting service of " << msgInServer->getName() << endl;

}

void Queue::putPacketInQueue(cMessage *msg)
{
    //queue.insert(msg);
    bst.insert(msg, exponential(avgServiceTime)); // add to tree
    emit(qlenSignal, bst.countNodesExcludingRoot());

    //log new message in queue
    EV << msg->getName() << " enters queue"<< endl;
}


