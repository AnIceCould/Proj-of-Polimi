#include <omnetpp.h>
#include <unordered_set>

using namespace omnetpp;

class Rand : public cSimpleModule
{
protected:
    cGate *inputGate;   // Input Gate
    cGate *outputGateOut;  // Output Gate "out"
    cGate *outputGateBack; // Output Gate "back"
    int packetCount; // Count the number of packets entering the module
    int outCount; // Count the number of packets going out from the out port
    double sojourns; //times in average a packet enters this
    std::unordered_set<cMessage*> backExitSet; // Record packets leaving from back
public:
    Rand();
    virtual ~Rand();

protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
};

Define_Module(Rand);

Rand::Rand(): packetCount(0), outCount(0) // Initialize packet count
{

}

Rand::~Rand()
{

}

void Rand::initialize()
{
    inputGate = gate("in");
    outputGateOut = gate("out");
    outputGateBack = gate("back");

    packetCount = 0;
    outCount = 0;
    WATCH(packetCount); // show packets in
    WATCH(outCount); // show packets out
    WATCH(sojourns); // times in average a packet enters this
}

void Rand::handleMessage(cMessage *msg)
{
    packetCount++;

    // back
    if (backExitSet.find(msg) != backExitSet.end()) {
        backExitSet.erase(msg);
        outCount++;
        send(msg, outputGateOut);
    } else {
        // random[0,1)
        double randomValue = uniform(0, 1);

        if (randomValue < 0.333) {
            // 33.3% "out"
            outCount++;
            send(msg, outputGateOut);
        } else {
            // to "back"
            backExitSet.insert(msg);
            send(msg, outputGateBack);
        }
    }
    if(outCount != 0)
        sojourns = (double)packetCount / outCount;
    else
        sojourns = 0;
}
