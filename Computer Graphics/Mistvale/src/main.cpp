// This has been adapted from the Vulkan tutorial
#include <sstream>

#include <json.hpp>

#include "modules/Starter.hpp"
#include "modules/TextMaker.hpp"
#include "modules/Scene.hpp"
#include "modules/Animations.hpp"

// The uniform buffer object used in this example
struct VertexChar {
	glm::vec3 pos;
	glm::vec3 norm;
	glm::vec2 UV;
	glm::uvec4 jointIndices;
	glm::vec4 weights;
};

struct VertexSimp {
	glm::vec3 pos;
	glm::vec3 norm;
	glm::vec2 UV;
};

struct skyBoxVertex {
	glm::vec3 pos;
};

struct VertexTan {
	glm::vec3 pos;
	glm::vec3 norm;
	glm::vec2 UV;
	glm::vec4 tan;
};

struct GlobalUniformBufferObject {
	alignas(16) glm::vec3 lightDir;
	alignas(16) glm::vec4 lightColor;
	alignas(16) glm::vec3 eyePos;
	alignas(16) glm::vec3 blueLightPos;
	alignas(16) glm::vec4 blueLightColor;
	alignas(16) glm::vec3 whiteLightPos;
	alignas(16) glm::vec4 whiteLightColor;
};

struct UniformBufferObjectChar {
	alignas(16) glm::vec4 debug1;
	alignas(16) glm::mat4 mvpMat[65];
	alignas(16) glm::mat4 mMat[65];
	alignas(16) glm::mat4 nMat[65];
};

struct UniformBufferObjectSimp {
	alignas(16) glm::mat4 mvpMat;
	alignas(16) glm::mat4 mMat;
	alignas(16) glm::mat4 nMat;
};

struct skyBoxUniformBufferObject {
	alignas(16) glm::mat4 mvpMat;
};




// MAIN ! 
class MISTVALE : public BaseProject {
	protected:
	// Here you list all the Vulkan objects you need:
	
	// Descriptor Layouts [what will be passed to the shaders]
	DescriptorSetLayout DSLlocalChar, DSLlocalSimp, DSLlocalPBR, DSLglobal, DSLskyBox;

	// Vertex formants, Pipelines [Shader couples] and Render passes
	VertexDescriptor VDchar;
	VertexDescriptor VDsimp;
	VertexDescriptor VDskyBox;
	VertexDescriptor VDtan;
	RenderPass RP;
	Pipeline Pchar, PsimpObj, PskyBox, P_PBR;
	//*DBG*/Pipeline PDebug;

	// Models, textures and Descriptors (values assigned to the uniforms)
	Scene SC;
	std::vector<VertexDescriptorRef>  VDRs;
	std::vector<TechniqueRef> PRs;
	//*DBG*/Model MS;
	//*DBG*/DescriptorSet SSD;
	
	// To support animation
	#define N_ANIMATIONS 1
	
	AnimBlender AB;
	Animations Anim[N_ANIMATIONS];
	SkeletalAnimation SKA;

	// to provide textual feedback
	TextMaker txt;
	
	// Other application parameters
	float Ar;	// Aspect ratio

	glm::mat4 ViewPrj;
	glm::mat4 World;
	glm::vec3 Pos = glm::vec3(0,0,5);
	glm::vec3 cameraPos;
	float Yaw = glm::radians(0.0f);
	float Pitch = glm::radians(0.0f);
	float Roll = glm::radians(0.0f);
	
	glm::vec4 debug1 = glm::vec4(0);
	bool redLightEnabled = true; // 控制红色聚光灯的开关状态
	
	// 外星人追逐者相关变量
	glm::vec3 alienPos = glm::vec3(0.0f, 0.0f, -50.0f); // 外星人初始位置
	float alienSpeed = 2.0f; // 外星人移动速度
	float alienRotationSpeed = 10.0f; // 外星人旋转速度
	float collisionDistance = 2.0f; // 碰撞检测距离
	bool gameOver = false; // 游戏结束标志
	// 计算外星人初始角度，让它面向玩家（玩家在[0,0,5]，外星人在[0,0,50]）
	float currentAlienAngle = glm::pi<float>(); // 初始面向玩家（朝南，需要180度偏移）
	
	// 游戏开始控制变量
	bool gameStarted = false; // 游戏是否已开始
	
	// WIN显示控制变量
	bool showWinText = false; // 是否显示WIN文本
	float winTextTimer = 0.0f; // WIN文本显示计时器
	
	// 道具收集系统相关变量
	int collectedItems = 0; // 已收集的道具数量
	const int totalItems = 3; // 总道具数量
	float itemCollectionDistance = 3.0f; // 道具收集距离
	bool itemsCollected[3] = {false, false, false}; // 道具收集状态
	
	// 道具聚光灯系统
	glm::vec3 itemSpotlightPos[3] = {
		glm::vec3(15.0f, 0.0f, -15.0f),  // 蓝色水晶位置
		glm::vec3(-15.0f, 0.0f, -25.0f), // 十字架位置
		glm::vec3(0.0f, 0.0f, -35.0f)    // 粗糙发光物体位置
	};
	float itemSpotlightIntensity[3] = {0.0f, 0.0f, 0.0f}; // 聚光灯强度
	const float spotlightRiseSpeed = 5.0f; // 聚光灯上升速度
	

	


	// Here you set the main application parameters
	void setWindowParameters() {
		// window size, titile and initial background
		windowWidth = 800;
		windowHeight = 600;
		windowTitle = "MISTVALE - Demon Hunter";
    	windowResizable = GLFW_TRUE;
		
		// Initial aspect ratio
		Ar = 4.0f / 3.0f;
	}
	
	// What to do when the window changes size
	void onWindowResize(int w, int h) {
		std::cout << "Window resized to: " << w << " x " << h << "\n";
		Ar = (float)w / (float)h;
		// Update Render Pass
		RP.width = w;
		RP.height = h;
		
		// updates the textual output
		txt.resizeScreen(w, h);
	}
	
	// Here you load and setup all your Vulkan Models and Texutures.
	// Here you also create your Descriptor set layouts and load the shaders for the pipelines
	void localInit() {
		// Descriptor Layouts [what will be passed to the shaders]
		DSLglobal.init(this, {
					// this array contains the binding:
					// first  element : the binding number
					// second element : the type of element (buffer or texture)
					// third  element : the pipeline stage where it will be used
					{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL_GRAPHICS, sizeof(GlobalUniformBufferObject), 1}
				  });

		DSLlocalChar.init(this, {
					// this array contains the binding:
					// first  element : the binding number
					// second element : the type of element (buffer or texture)
					// third  element : the pipeline stage where it will be used
					{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, sizeof(UniformBufferObjectChar), 1},
					{1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 1}
				  });

		DSLlocalSimp.init(this, {
					// this array contains the binding:
					// first  element : the binding number
					// second element : the type of element (buffer or texture)
					// third  element : the pipeline stage where it will be used
					{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, sizeof(UniformBufferObjectSimp), 1},
					{1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 1},
					{2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1, 1}
				  });

		DSLskyBox.init(this, {
			{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, sizeof(skyBoxUniformBufferObject), 1},
			{1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 1}
		  });

		DSLlocalPBR.init(this, {
					// this array contains the binding:
					// first  element : the binding number
					// second element : the type of element (buffer or texture)
					// third  element : the pipeline stage where it will be used
					{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, sizeof(UniformBufferObjectSimp), 1},
					{1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 1},
					{2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1, 1},
					{3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 2, 1},
                    {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 3, 1}
				  });

		VDchar.init(this, {
				  {0, sizeof(VertexChar), VK_VERTEX_INPUT_RATE_VERTEX}
				}, {
				  {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexChar, pos),
				         sizeof(glm::vec3), POSITION},
				  {0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexChar, norm),
				         sizeof(glm::vec3), NORMAL},
				  {0, 2, VK_FORMAT_R32G32_SFLOAT, offsetof(VertexChar, UV),
				         sizeof(glm::vec2), UV},
					{0, 3, VK_FORMAT_R32G32B32A32_UINT, offsetof(VertexChar, jointIndices),
				         sizeof(glm::uvec4), JOINTINDEX},
					{0, 4, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(VertexChar, weights),
				         sizeof(glm::vec4), JOINTWEIGHT}
				});

		VDsimp.init(this, {
				  {0, sizeof(VertexSimp), VK_VERTEX_INPUT_RATE_VERTEX}
				}, {
				  {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexSimp, pos),
				         sizeof(glm::vec3), POSITION},
				  {0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexSimp, norm),
				         sizeof(glm::vec3), NORMAL},
				  {0, 2, VK_FORMAT_R32G32_SFLOAT, offsetof(VertexSimp, UV),
				         sizeof(glm::vec2), UV}
				});

		VDskyBox.init(this, {
		  {0, sizeof(skyBoxVertex), VK_VERTEX_INPUT_RATE_VERTEX}
		}, {
		  {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(skyBoxVertex, pos),
				 sizeof(glm::vec3), POSITION}
		});

		VDtan.init(this, {
				  {0, sizeof(VertexTan), VK_VERTEX_INPUT_RATE_VERTEX}
				}, {
				  {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexTan, pos),
				         sizeof(glm::vec3), POSITION},
				  {0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexTan, norm),
				         sizeof(glm::vec3), NORMAL},
				  {0, 2, VK_FORMAT_R32G32_SFLOAT, offsetof(VertexTan, UV),
				         sizeof(glm::vec2), UV},
				  {0, 3, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(VertexTan, tan),
				         sizeof(glm::vec4), TANGENT}
				});
				
		VDRs.resize(4);
		VDRs[0].init("VDchar",   &VDchar);
		VDRs[1].init("VDsimp",   &VDsimp);
		VDRs[2].init("VDskybox", &VDskyBox);
		VDRs[3].init("VDtan",    &VDtan);
		
		// initializes the render passes
		RP.init(this);
		// sets the blue sky
		RP.properties[0].clearValue = {0.0f,0.9f,1.0f,1.0f};
		

		// Pipelines [Shader couples]
		// The last array, is a vector of pointer to the layouts of the sets that will
		// be used in this pipeline. The first element will be set 0, and so on..
		Pchar.init(this, &VDchar, "shaders/PosNormUvTanWeights.vert.spv", "shaders/CookTorranceForCharacter.frag.spv", {&DSLglobal, &DSLlocalChar});

		PsimpObj.init(this, &VDsimp, "shaders/SimplePosNormUV.vert.spv", "shaders/CookTorrance.frag.spv", {&DSLglobal, &DSLlocalSimp});

		PskyBox.init(this, &VDskyBox, "shaders/SkyBoxShader.vert.spv", "shaders/SkyBoxShader.frag.spv", {&DSLskyBox});
		PskyBox.setCompareOp(VK_COMPARE_OP_LESS_OR_EQUAL);
		PskyBox.setCullMode(VK_CULL_MODE_BACK_BIT);
		PskyBox.setPolygonMode(VK_POLYGON_MODE_FILL);

		P_PBR.init(this, &VDtan, "shaders/SimplePosNormUvTan.vert.spv", "shaders/PBR.frag.spv", {&DSLglobal, &DSLlocalPBR});

		PRs.resize(4);
		PRs[0].init("CookTorranceChar", {
							 {&Pchar, {//Pipeline and DSL for the first pass
								 /*DSLglobal*/{},
								 /*DSLlocalChar*/{
										/*t0*/{true,  0, {}}// index 0 of the "texture" field in the json file
									 }
									}}
							  }, /*TotalNtextures*/1, &VDchar);
		PRs[1].init("CookTorranceNoiseSimp", {
							 {&PsimpObj, {//Pipeline and DSL for the first pass
								 /*DSLglobal*/{},
								 /*DSLlocalSimp*/{
										/*t0*/{true,  0, {}},// index 0 of the "texture" field in the json file
										/*t1*/{true,  1, {}} // index 1 of the "texture" field in the json file
									 }
									}}
							  }, /*TotalNtextures*/2, &VDsimp);
		PRs[2].init("SkyBox", {
							 {&PskyBox, {//Pipeline and DSL for the first pass
								 /*DSLskyBox*/{
										/*t0*/{true,  0, {}}// index 0 of the "texture" field in the json file
									 }
									}}
							  }, /*TotalNtextures*/1, &VDskyBox);
		PRs[3].init("PBR", {
							 {&P_PBR, {//Pipeline and DSL for the first pass
								 /*DSLglobal*/{},
								 /*DSLlocalPBR*/{
										/*t0*/{true,  0, {}},// index 0 of the "texture" field in the json file
										/*t1*/{true,  1, {}},// index 1 of the "texture" field in the json file
										/*t2*/{true,  2, {}},// index 2 of the "texture" field in the json file
										/*t3*/{true,  3, {}}// index 3 of the "texture" field in the json file
									 }
									}}
							  }, /*TotalNtextures*/4, &VDtan);

		// Models, textures and Descriptors (values assigned to the uniforms)
		
		// sets the size of the Descriptor Set Pool
		DPSZs.uniformBlocksInPool = 3;
		DPSZs.texturesInPool = 4;
		DPSZs.setsInPool = 3;
		
std::cout << "\nLoading the scene\n\n";
		if(SC.init(this, /*Npasses*/1, VDRs, PRs, "assets/models/scene.json") != 0) {
			std::cout << "ERROR LOADING THE SCENE\n";
			exit(0);
		}
		// initializes animation (only one)
		Anim[0].init(*SC.As[0]);
		AB.init({{0,32,0.0f,0}, {0,16,0.0f,1}, {0,263,0.0f,2}, {0,83,0.0f,3}, {0,16,0.0f,4}});
		SKA.init(Anim, 1, "Armature|mixamo.com|Layer0", 0);
		
		// initializes the textual output
		txt.init(this, windowWidth, windowHeight);

		// submits the main command buffer
		submitCommandBuffer("main", 0, populateCommandBufferAccess, this);

		// Prepares for showing the FPS count
		txt.print(1.0f, 1.0f, "FPS:",1,"CO",false,false,true,TAL_RIGHT,TRH_RIGHT,TRV_BOTTOM,{1.0f,0.0f,0.0f,1.0f},{0.8f,0.8f,0.0f,1.0f});
		
		// 添加聚光灯控制提示
		txt.print(1.0f, 1.0f, "Press 5 to toggle red spotlight",1,"CO",false,false,true,TAL_LEFT,TRH_LEFT,TRV_TOP,{1.0f,1.0f,1.0f,1.0f},{0.0f,0.0f,0.0f,1.0f});
	}
	
	// Here you create your pipelines and Descriptor Sets!
	void pipelinesAndDescriptorSetsInit() {
		// creates the render pass
		RP.create();
		
		// This creates a new pipeline (with the current surface), using its shaders for the provided render pass
		Pchar.create(&RP);
		PsimpObj.create(&RP);
		PskyBox.create(&RP);
		P_PBR.create(&RP);
		
		SC.pipelinesAndDescriptorSetsInit();
		txt.pipelinesAndDescriptorSetsInit();
	}

	// Here you destroy your pipelines and Descriptor Sets!
	void pipelinesAndDescriptorSetsCleanup() {
		Pchar.cleanup();
		PsimpObj.cleanup();
		PskyBox.cleanup();
		P_PBR.cleanup();
		RP.cleanup();

		SC.pipelinesAndDescriptorSetsCleanup();
		txt.pipelinesAndDescriptorSetsCleanup();
	}

	// Here you destroy all the Models, Texture and Desc. Set Layouts you created!
	// You also have to destroy the pipelines
	void localCleanup() {
		DSLlocalChar.cleanup();
		DSLlocalSimp.cleanup();
		DSLlocalPBR.cleanup();
		DSLskyBox.cleanup();
		DSLglobal.cleanup();
		
		Pchar.destroy();	
		PsimpObj.destroy();
		PskyBox.destroy();		
		P_PBR.destroy();		

		RP.destroy();

		SC.localCleanup();	
		txt.localCleanup();
		
		Anim[0].cleanup();
	}
	
	// Here it is the creation of the command buffer:
	// You send to the GPU all the objects you want to draw,
	// with their buffers and textures
	static void populateCommandBufferAccess(VkCommandBuffer commandBuffer, int currentImage, void *Params) {
		// Simple trick to avoid having always 'T->'
		// in che code that populates the command buffer!
//std::cout << "Populating command buffer for " << currentImage << "\n";
		MISTVALE *T = (MISTVALE *)Params;
		T->populateCommandBuffer(commandBuffer, currentImage);
	}
	// This is the real place where the Command Buffer is written
	void populateCommandBuffer(VkCommandBuffer commandBuffer, int currentImage) {
		
		// begin standard pass
		RP.begin(commandBuffer, currentImage);

		SC.populateCommandBuffer(commandBuffer, 0, currentImage);

		RP.end(commandBuffer);
	}

	// Here is where you update the uniforms.
	// Very likely this will be where you will be writing the logic of your application.
	void updateUniformBuffer(uint32_t currentImage) {
		static bool debounce = false;
		static int curDebounce = 0;
		
		// handle the ESC key to exit the app
		if(glfwGetKey(window, GLFW_KEY_ESCAPE)) {
			glfwSetWindowShouldClose(window, GL_TRUE);
		}


		if(glfwGetKey(window, GLFW_KEY_1)) {
			if(!debounce) {
				debounce = true;
				curDebounce = GLFW_KEY_1;

				debug1.x = 1.0 - debug1.x;
			}
		} else {
			if((curDebounce == GLFW_KEY_1) && debounce) {
				debounce = false;
				curDebounce = 0;
			}
		}

		if(glfwGetKey(window, GLFW_KEY_2)) {
			if(!debounce) {
				debounce = true;
				curDebounce = GLFW_KEY_2;

				debug1.y = 1.0 - debug1.y;
			}
		} else {
			if((curDebounce == GLFW_KEY_2) && debounce) {
				debounce = false;
				curDebounce = 0;
			}
		}

		if(glfwGetKey(window, GLFW_KEY_P)) {
			if(!debounce) {
				debounce = true;
				curDebounce = GLFW_KEY_P;

				debug1.z = (float)(((int)debug1.z + 1) % 65);
std::cout << "Showing bone index: " << debug1.z << "\n";
			}
		} else {
			if((curDebounce == GLFW_KEY_P) && debounce) {
				debounce = false;
				curDebounce = 0;
			}
		}

		if(glfwGetKey(window, GLFW_KEY_O)) {
			if(!debounce) {
				debounce = true;
				curDebounce = GLFW_KEY_O;

				debug1.z = (float)(((int)debug1.z + 64) % 65);
std::cout << "Showing bone index: " << debug1.z << "\n";
			}
		} else {
			if((curDebounce == GLFW_KEY_O) && debounce) {
				debounce = false;
				curDebounce = 0;
			}
		}

		// 按键5控制红色聚光灯开关
		if(glfwGetKey(window, GLFW_KEY_5)) {
			if(!debounce) {
				debounce = true;
				curDebounce = GLFW_KEY_5;

				redLightEnabled = !redLightEnabled; // 切换聚光灯状态
				std::cout << "Red spotlight " << (redLightEnabled ? "ENABLED" : "DISABLED") << "\n";
			}
		} else {
			if((curDebounce == GLFW_KEY_5) && debounce) {
				debounce = false;
				curDebounce = 0;
			}
		}
		
		// 按键R重置游戏
		if(glfwGetKey(window, GLFW_KEY_E)) {
			if(!debounce) {
				debounce = true;
				curDebounce = GLFW_KEY_R;

				// 重置游戏状态
				gameOver = false;
				gameStarted = false; // 重新显示开始提示
				showWinText = false; // 重置WIN文本显示
				winTextTimer = 0.0f; // 重置WIN文本计时器
				alienPos = glm::vec3(0.0f, 0.0f, 50.0f); // 重置外星人位置
				// 重置外星人角度，让它面向玩家（玩家在[0,0,5]，外星人在[0,0,50]）
				currentAlienAngle = glm::pi<float>(); // 重置面向玩家（朝南，需要180度偏移）
				// 重置道具收集状态
				collectedItems = 0;
				for(int i = 0; i < 3; i++) {
					itemsCollected[i] = false;
					itemSpotlightIntensity[i] = 0.0f; // 重置聚光灯强度
				}
				std::cout << "Game reset! Press ENTER to start again." << "\n";
			}
		} else {
			if((curDebounce == GLFW_KEY_R) && debounce) {
				debounce = false;
				curDebounce = 0;
			}
		}
		
		// 按键Enter开始游戏
		if(glfwGetKey(window, GLFW_KEY_ENTER)) {
			if(!debounce) {
				debounce = true;
				curDebounce = GLFW_KEY_ENTER;

				gameStarted = true; // 开始游戏
				std::cout << "Game started!" << "\n";
			}
		} else {
			if((curDebounce == GLFW_KEY_ENTER) && debounce) {
				debounce = false;
				curDebounce = 0;
			}
		}
		


		// moves the view
		float deltaT = GameLogic();
		
		// 如果游戏还没开始，禁用玩家移动
		if (!gameStarted) {
			// 重置玩家位置到起始位置
			Pos = glm::vec3(0.0, 0.0, 5);
			Yaw = glm::radians(0.0f);
			Pitch = glm::radians(0.0f);
		}
		
		// updated the animation
		const float SpeedUpAnimFact = 0.85f;
		AB.Advance(deltaT * SpeedUpAnimFact);
		
		// defines the global parameters for the uniform
		// 设置光照方向 - 白天光照：太阳从上方照射，角度为30度和45度
		const glm::mat4 lightView = glm::rotate(glm::mat4(1), glm::radians(30.0f), glm::vec3(0.0f,1.0f,0.0f)) * glm::rotate(glm::mat4(1), glm::radians(45.0f), glm::vec3(1.0f,0.0f,0.0f));
		const glm::vec3 lightDir = glm::vec3(lightView * glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));
	
		GlobalUniformBufferObject gubo{};

		gubo.lightDir = lightDir;
		// 设置光照颜色 - 降低全局光照强度，创造更暗的环境
		gubo.lightColor = glm::vec4(0.6f, 0.55f, 0.45f, 1.0f);
		gubo.eyePos = cameraPos;
		
		// 设置红色聚光灯 - 位置在[0,0,-10]，从上到下的红色聚光灯
		gubo.blueLightPos = glm::vec3(0.0f, 25.0f, -10.0f); // 位置在[0,0,-10]上方，更高一些
		// 根据开关状态设置聚光灯强度
		float redLightIntensity = redLightEnabled ? 15.0f : 0.0f;
		gubo.blueLightColor = glm::vec4(1.0f, 0.2f, 0.2f, redLightIntensity); // 可控制的红色聚光灯
		
		// 设置白色聚光灯 - 位置在[20,25,-2]，从上到下的白色聚光灯
		gubo.whiteLightPos = glm::vec3(20.0f, 25.0f, -2.0f); // 奖杯上方25个单位
		gubo.whiteLightColor = glm::vec4(1.0f, 1.0f, 1.0f, 15.0f); // 白色聚光灯，始终开启
		
		// 添加道具聚光灯效果 - 使用额外的光照通道
		// 这里我们通过修改全局光照来模拟道具聚光灯
		// 如果有多个道具被收集，聚光灯效果会叠加
		float totalItemSpotlight = 0.0f;
		for(int i = 0; i < 3; i++) {
			if (itemsCollected[i]) {
				totalItemSpotlight += itemSpotlightIntensity[i];
			}
		}
		
		// 将道具聚光灯效果添加到全局光照中
		if (totalItemSpotlight > 0.0f) {
			// 增强全局光照来模拟道具聚光灯效果
			gubo.lightColor += glm::vec4(0.3f, 0.5f, 1.0f, 0.0f) * (totalItemSpotlight / 20.0f);
		}
		
		// 为已收集的道具添加特殊光照效果
		// 这里我们通过修改全局光照来模拟道具下方的聚光灯
		for(int i = 0; i < 3; i++) {
			if (itemsCollected[i] && itemSpotlightIntensity[i] > 0.0f) {
				// 在道具位置添加向上的聚光灯效果
				glm::vec3 spotlightPos = itemSpotlightPos[i];
				float distanceToSpotlight = glm::length(Pos - spotlightPos);
				float spotlightEffect = itemSpotlightIntensity[i] * (1.0f / (1.0f + distanceToSpotlight * 0.1f));
				
				// 根据道具类型设置不同的聚光灯颜色
				glm::vec4 spotlightColor;
				switch(i) {
					case 0: // 水晶 - 蓝色
						spotlightColor = glm::vec4(0.2f, 0.6f, 1.0f, 0.0f);
						break;
					case 1: // 十字架 - 红色
						spotlightColor = glm::vec4(1.0f, 0.2f, 0.2f, 0.0f);
						break;
					case 2: // 烛台 - 绿色
						spotlightColor = glm::vec4(0.2f, 1.0f, 0.3f, 0.0f);
						break;
					default:
						spotlightColor = glm::vec4(0.2f, 0.8f, 1.0f, 0.0f);
						break;
				}
				
				// 增强全局光照来模拟聚光灯
				gubo.lightColor += spotlightColor * spotlightEffect;
			}
		}
		

		


		// defines the local parameters for the uniforms
		UniformBufferObjectChar uboc{};	
		uboc.debug1 = debug1;

		SKA.Sample(AB);
		std::vector<glm::mat4> *TMsp = SKA.getTransformMatrices();
		
//printMat4("TF[55]", (*TMsp)[55]);
		
		glm::mat4 AdaptMat =
			glm::scale(glm::mat4(1.0f), glm::vec3(0.01f)) *
			glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f,0.0f,0.0f));

		int instanceId;
		// character
		for(instanceId = 0; instanceId < SC.TI[0].InstanceCount; instanceId++) {
			for(int im = 0; im < TMsp->size(); im++) {
				uboc.mMat[im]   = World * AdaptMat * (*TMsp)[im];
				uboc.mvpMat[im] = ViewPrj * uboc.mMat[im];
				uboc.nMat[im] = glm::inverse(glm::transpose(uboc.mMat[im]));
				//std::cout << im << "\t";
				//printMat4("mMat", ubo.mMat[im]);
			}

			SC.TI[0].I[instanceId].DS[0][0]->map(currentImage, &gubo, 0); // Set 0
			SC.TI[0].I[instanceId].DS[0][1]->map(currentImage, &uboc, 0);  // Set 1
		}

		UniformBufferObjectSimp ubos{};	
		// normal objects
		for(instanceId = 0; instanceId < SC.TI[1].InstanceCount; instanceId++) {
			std::string instanceIdStr = *SC.TI[1].I[instanceId].id;
			
			// 检查是否是外星人实例
			if (instanceIdStr == "alien_warlord_instance") {
				// 让ghost的脸始终朝向玩家
				glm::vec3 toPlayer = Pos - alienPos;
				float targetAngle = atan2(toPlayer.x, toPlayer.z) + glm::pi<float>(); // 目标旋转角度
				
				// 直接旋转，不使用平滑插值
				currentAlienAngle = targetAngle;
				
				ubos.mMat = glm::translate(glm::mat4(1.0f), alienPos + glm::vec3(0, 1.5f, 0)) *
							glm::rotate(glm::mat4(1.0f), currentAlienAngle, glm::vec3(0, 1, 0)) *
							glm::scale(glm::mat4(1.0f), glm::vec3(3.0f));
				
				ubos.mvpMat = ViewPrj * ubos.mMat;
				ubos.nMat   = glm::inverse(glm::transpose(ubos.mMat));

				SC.TI[1].I[instanceId].DS[0][0]->map(currentImage, &gubo, 0); // Set 0
				SC.TI[1].I[instanceId].DS[0][1]->map(currentImage, &ubos, 0);  // Set 1
			}

			// 道具实例 - 始终渲染，但收集后会有聚光灯效果
			else if (instanceIdStr == "crystal_blue_instance") {
				ubos.mMat = SC.TI[1].I[instanceId].Wm;
				ubos.mvpMat = ViewPrj * ubos.mMat;
				ubos.nMat   = glm::inverse(glm::transpose(ubos.mMat));

				SC.TI[1].I[instanceId].DS[0][0]->map(currentImage, &gubo, 0); // Set 0
				SC.TI[1].I[instanceId].DS[0][1]->map(currentImage, &ubos, 0);  // Set 1
			}
			else if (instanceIdStr == "ethereal_crucifixion_instance") {
				ubos.mMat = SC.TI[1].I[instanceId].Wm;
				ubos.mvpMat = ViewPrj * ubos.mMat;
				ubos.nMat   = glm::inverse(glm::transpose(ubos.mMat));

				SC.TI[1].I[instanceId].DS[0][0]->map(currentImage, &gubo, 0); // Set 0
				SC.TI[1].I[instanceId].DS[0][1]->map(currentImage, &ubos, 0);  // Set 1
			}
			else if (instanceIdStr == "rough_lighted_instance") {
				ubos.mMat = SC.TI[1].I[instanceId].Wm;
				ubos.mvpMat = ViewPrj * ubos.mMat;
				ubos.nMat   = glm::inverse(glm::transpose(ubos.mMat));

				SC.TI[1].I[instanceId].DS[0][0]->map(currentImage, &gubo, 0); // Set 0
				SC.TI[1].I[instanceId].DS[0][1]->map(currentImage, &ubos, 0);  // Set 1
			}
			else if (instanceIdStr == "golden_trophy_instance") {
				ubos.mMat = SC.TI[1].I[instanceId].Wm;
				ubos.mvpMat = ViewPrj * ubos.mMat;
				ubos.nMat   = glm::inverse(glm::transpose(ubos.mMat));

				SC.TI[1].I[instanceId].DS[0][0]->map(currentImage, &gubo, 0); // Set 0
				SC.TI[1].I[instanceId].DS[0][1]->map(currentImage, &ubos, 0);  // Set 1
			}
			// 其他非道具对象
			else {
				ubos.mMat = SC.TI[1].I[instanceId].Wm;
				ubos.mvpMat = ViewPrj * ubos.mMat;
				ubos.nMat   = glm::inverse(glm::transpose(ubos.mMat));

				SC.TI[1].I[instanceId].DS[0][0]->map(currentImage, &gubo, 0); // Set 0
				SC.TI[1].I[instanceId].DS[0][1]->map(currentImage, &ubos, 0);  // Set 1
			}
		}
		
		// skybox pipeline
		skyBoxUniformBufferObject sbubo{};
		sbubo.mvpMat = ViewPrj * glm::translate(glm::mat4(1), cameraPos) * glm::scale(glm::mat4(1), glm::vec3(100.0f));
		SC.TI[2].I[0].DS[0][0]->map(currentImage, &sbubo, 0);

		// PBR objects
		for(instanceId = 0; instanceId < SC.TI[3].InstanceCount; instanceId++) {
			ubos.mMat   = SC.TI[3].I[instanceId].Wm;
			ubos.mvpMat = ViewPrj * ubos.mMat;
			ubos.nMat   = glm::inverse(glm::transpose(ubos.mMat));

			SC.TI[3].I[instanceId].DS[0][0]->map(currentImage, &gubo, 0); // Set 0
			SC.TI[3].I[instanceId].DS[0][1]->map(currentImage, &ubos, 0);  // Set 1
		}


		// updates the FPS
		static float elapsedT = 0.0f;
		static int countedFrames = 0;
		
		countedFrames++;
		elapsedT += deltaT;
		if(elapsedT > 1.0f) {
			float Fps = (float)countedFrames / elapsedT;
			
			std::ostringstream oss;
			oss << "FPS: " << Fps << "\n";

			txt.print(1.0f, 1.0f, oss.str(), 1, "CO", false, false, true,TAL_RIGHT,TRH_RIGHT,TRV_BOTTOM,{1.0f,0.0f,0.0f,1.0f},{0.8f,0.8f,0.0f,1.0f});
			
			elapsedT = 0.0f;
		    countedFrames = 0;
		}
		
		// Update spotlight status text
		static int statusTextId = -1;
		if(statusTextId == -1) {
			statusTextId = txt.print(1.0f, 1.0f, "", 2, "CO", false, false, true, TAL_LEFT, TRH_LEFT, TRV_TOP, {1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f});
		}
		
		// Update status text
		std::string lightStatus = "Red Spotlight: " + std::string(redLightEnabled ? "ON" : "OFF");
		txt.removeText(statusTextId);
		statusTextId = txt.print(1.0f, 1.0f, lightStatus.c_str(), 2, "CO", false, false, true, TAL_LEFT, TRH_LEFT, TRV_TOP, {1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f});
		
		// 游戏说明文本
		static int instructionTextId = -1;
		if (instructionTextId == -1) {
			instructionTextId = txt.print(1.0f, 1.0f, "Controls: WASD=Move, Mouse=Look, 5=Toggle Spotlight, E=Reset Game, T=Teleport, ENTER=Start", 2, "CO", false, false, true, TAL_LEFT, TRH_LEFT, TRV_BOTTOM, {1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f});
		}
		
		// 游戏背景故事文本
		static int storyTextId = -1;
		if (!gameStarted) {
			if (storyTextId == -1) {
				storyTextId = txt.print(0.0f, 0.0f, "MISTVALE\n\nThe town of Mistvale is shrouded in perpetual gloom.\n\n An ancient clock tower stands in the center, said to control the cycle of day and night.\n\n But now its chimes have ceased, and the town is plagued by strange occurrences—vanishing townsfolk, ghosts roaming the night, whispers from beneath the church.\n\nYou are a demon hunter. Collect crystals and crucifixes, and defeat the fiends.", 12, "SS", false, false, false, TAL_CENTER, TRH_CENTER, TRV_MIDDLE, {1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f});
			}
		} else if (storyTextId != -1) {
			txt.removeText(storyTextId);
			storyTextId = -1;
		}
		
		// 游戏开始提示文本
		static int startGameTextId = -1;
		if (!gameStarted) {
			if (startGameTextId == -1) {
				startGameTextId = txt.print(0.0f, 0.9f, "Press ENTER to start the game", 15, "SS", false, false, false, TAL_CENTER, TRH_CENTER, TRV_MIDDLE, {1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f});
			}
		} else if (startGameTextId != -1) {
			txt.removeText(startGameTextId);
			startGameTextId = -1;
		}
		
		// WIN文本显示
		static int winTextId = -1;
		if (showWinText && winTextTimer > 0.0f) {
			if (winTextId == -1) {
				winTextId = txt.print(0.0f, 0.0f, "WIN!", 25, "SS", false, false, false, TAL_CENTER, TRH_CENTER, TRV_MIDDLE, {0.0f, 1.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f});
			}
			winTextTimer -= deltaT; // 减少计时器
			if (winTextTimer <= 0.0f) {
				showWinText = false;
				winTextTimer = 0.0f;
			}
		} else if (winTextId != -1) {
			txt.removeText(winTextId);
			winTextId = -1;
		}
		
		// 道具收集数量显示
		static int itemCountTextId = -1;
		if (itemCountTextId == -1) {
			itemCountTextId = txt.print(1.0f, 1.0f, "", 4, "CO", false, false, true, TAL_CENTER, TRH_CENTER, TRV_BOTTOM, {0.0f, 1.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f});
		}
		
		// 更新道具收集数量文本
		std::ostringstream itemCountOss;
		itemCountOss << "Items Collected: " << collectedItems << "/" << totalItems;
		if (collectedItems > 0) {
			itemCountOss << " (Spotlights Active!)";
		}
		txt.removeText(itemCountTextId);
		itemCountTextId = txt.print(1.0f, 1.0f, itemCountOss.str().c_str(), 4, "CO", false, false, true, TAL_CENTER, TRH_CENTER, TRV_BOTTOM, {0.0f, 1.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f});
		
		// 显示道具颜色说明
		if (collectedItems > 0) {
			static int colorInfoTextId = -1;
			if (colorInfoTextId == -1) {
				colorInfoTextId = txt.print(1.0f, 0.9f, "Crystal: Blue | Cross: Red | Candle: Green", 3, "CO", false, false, true, TAL_CENTER, TRH_CENTER, TRV_BOTTOM, {0.8f, 0.8f, 0.8f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f});
			}
		} else {
			static int colorInfoTextId = -1;
			if (colorInfoTextId != -1) {
				txt.removeText(colorInfoTextId);
				colorInfoTextId = -1;
			}
		}
		
		// 游戏失败提示
		static int gameOverTextId = -1;
		if (gameOver) {
			if (gameOverTextId == -1) {
				gameOverTextId = txt.print(0.0f, 0.0f, "GAME OVER! Ghost caught you!", 20, "SS", false, false, false, TAL_CENTER, TRH_CENTER, TRV_MIDDLE, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f});
			}
		} else if (gameOverTextId != -1) {
			txt.removeText(gameOverTextId);
			gameOverTextId = -1;
		}
		
		txt.updateCommandBuffer();
	}
	
	float GameLogic() {
		// Parameters
		// Camera FOV-y, Near Plane and Far Plane
		const float FOVy = glm::radians(45.0f);
		const float nearPlane = 0.1f;
		const float farPlane = 100.f;
		// Player starting point
		const glm::vec3 StartingPosition = glm::vec3(0.0, 0.0, 5);
		// Camera target height and distance
		static float camHeight = 1.5;
		static float camDist = 5;
		// Camera Pitch limits
		const float minPitch = glm::radians(-8.75f);
		const float maxPitch = glm::radians(60.0f);
		// Rotation and motion speed
		const float ROT_SPEED = glm::radians(120.0f);
		const float MOVE_SPEED_BASE = 2.0f;
		const float MOVE_SPEED_RUN  = 5.0f;
		const float ZOOM_SPEED = MOVE_SPEED_BASE * 1.5f;
		const float MAX_CAM_DIST =  7.5;
		const float MIN_CAM_DIST =  1.5;

		// Integration with the timers and the controllers
		float deltaT;
		glm::vec3 m = glm::vec3(0.0f), r = glm::vec3(0.0f);
		bool fire = false;
		getSixAxis(deltaT, m, r, fire);
		float MOVE_SPEED = fire ? MOVE_SPEED_RUN : MOVE_SPEED_BASE;
		
		// 如果游戏还没开始，禁用移动和旋转
		if (!gameStarted) {
			m = glm::vec3(0.0f);
			r = glm::vec3(0.0f);
		}


		// Game Logic implementation
		// Current Player Position - statc variable make sure its value remain unchanged in subsequent calls to the procedure
		static glm::vec3 Pos = StartingPosition;
		static glm::vec3 oldPos;
		static int currRunState = 1;
		
		// 传送功能检测
		static bool teleportDebounce = false;
		if(glfwGetKey(window, GLFW_KEY_T)) {
			if(!teleportDebounce) {
				teleportDebounce = true;
				// 传送到指定位置
				Pos = glm::vec3(20.0f, 0.0f, 0.0f);
				// 显示WIN文本
				showWinText = true;
				winTextTimer = 3.0f; // 显示3秒
				std::cout << "Teleported to [20, 0, 0] - WIN!" << "\n";
			}
		} else {
			teleportDebounce = false;
		}

/*		camDist = camDist - m.y * ZOOM_SPEED * deltaT;
		camDist = camDist < MIN_CAM_DIST ? MIN_CAM_DIST :
				 (camDist > MAX_CAM_DIST ? MAX_CAM_DIST : camDist);*/
		camDist = (MIN_CAM_DIST + MIN_CAM_DIST) / 2.0f; 

		// To be done in the assignment
		ViewPrj = glm::mat4(1);
		World = glm::mat4(1);

		oldPos = Pos;

		static float Yaw = glm::radians(0.0f);
		static float Pitch = glm::radians(0.0f);
		static float relDir = glm::radians(0.0f);
		static float dampedRelDir = glm::radians(0.0f);
		static glm::vec3 dampedCamPos = StartingPosition;
		
		// World
		// Position
		glm::vec3 ux = glm::rotate(glm::mat4(1.0f), Yaw, glm::vec3(0,1,0)) * glm::vec4(1,0,0,1);
		glm::vec3 uz = glm::rotate(glm::mat4(1.0f), Yaw, glm::vec3(0,1,0)) * glm::vec4(0,0,-1,1);
		Pos = Pos + MOVE_SPEED * m.x * ux * deltaT;
		Pos = Pos - MOVE_SPEED * m.z * uz * deltaT;
		
		camHeight += MOVE_SPEED * m.y * deltaT;
		// Rotation
		Yaw = Yaw - ROT_SPEED * deltaT * r.y;
		Pitch = Pitch - ROT_SPEED * deltaT * r.x;
		Pitch  =  Pitch < minPitch ? minPitch :
				   (Pitch > maxPitch ? maxPitch : Pitch);


		float ef = exp(-10.0 * deltaT);
		// Rotational independence from view with damping
		if(glm::length(glm::vec3(m.x, 0.0f, m.z)) > 0.001f) {
			relDir = Yaw + atan2(m.x, m.z);
			dampedRelDir = dampedRelDir > relDir + 3.1416f ? dampedRelDir - 6.28f :
						   dampedRelDir < relDir - 3.1416f ? dampedRelDir + 6.28f : dampedRelDir;
		}
		dampedRelDir = ef * dampedRelDir + (1.0f - ef) * relDir;
		
		// Final world matrix computaiton
		float yOffset = 1.0f; // 人物整体上升 2 个单位
		float characterScale = 0.5f; // 人物尺寸缩小到原来的一半
		World = glm::translate(glm::mat4(1), Pos + glm::vec3(0, yOffset, 0)) * glm::rotate(glm::mat4(1.0f), dampedRelDir, glm::vec3(0,1,0)) * glm::scale(glm::mat4(1.0f), glm::vec3(characterScale));
		
		// Projection
		glm::mat4 Prj = glm::perspective(FOVy, Ar, nearPlane, farPlane);
		Prj[1][1] *= -1;

		// View
		// Target
		glm::vec3 target = Pos + glm::vec3(0.0f, camHeight, 0.0f);

		// Camera position, depending on Yaw parameter, but not character direction
		glm::mat4 camWorld = glm::translate(glm::mat4(1), Pos) * glm::rotate(glm::mat4(1.0f), Yaw, glm::vec3(0,1,0));
		cameraPos = camWorld * glm::vec4(0.0f, camHeight + camDist * sin(Pitch), camDist * cos(Pitch), 1.0);
		// Damping of camera
		dampedCamPos = ef * dampedCamPos + (1.0f - ef) * cameraPos;

		glm::mat4 View = glm::lookAt(dampedCamPos, target, glm::vec3(0,1,0));

		ViewPrj = Prj * View;
		
		float vel = length(Pos - oldPos) / deltaT;
		
		// 外星人追逐逻辑
		if (!gameOver && gameStarted) {
			glm::vec3 directionToPlayer = glm::normalize(Pos - alienPos);
			alienPos += directionToPlayer * alienSpeed * deltaT;
			
			// 碰撞检测
			float distanceToPlayer = glm::length(Pos - alienPos);
			if (distanceToPlayer < collisionDistance) {
				gameOver = true;
			}
		}
		
		// 道具收集检测逻辑
		if (!gameOver && gameStarted) {
			// 道具1: 蓝色水晶 [15, 2, -15]
			if (!itemsCollected[0]) {
				glm::vec3 itemPos1 = glm::vec3(15.0f, 2.0f, -15.0f);
				float distanceToItem1 = glm::length(Pos - itemPos1);
				if (distanceToItem1 < itemCollectionDistance) {
					itemsCollected[0] = true;
					collectedItems++;
					itemSpotlightIntensity[0] = 0.0f; // 开始聚光灯效果
					std::cout << "Collected Crystal Blue! Total: " << collectedItems << "/" << totalItems << std::endl;
				}
			}
			
			// 道具2: 十字架 [-15, 2, -25]
			if (!itemsCollected[1]) {
				glm::vec3 itemPos2 = glm::vec3(-15.0f, 2.0f, -25.0f);
				float distanceToItem2 = glm::length(Pos - itemPos2);
				if (distanceToItem2 < itemCollectionDistance) {
					itemsCollected[1] = true;
					collectedItems++;
					itemSpotlightIntensity[1] = 0.0f; // 开始聚光灯效果
					std::cout << "Collected Ethereal Crucifixion! Total: " << collectedItems << "/" << totalItems << std::endl;
				}
			}
			
			// 道具3: 粗糙发光物体 [0, 2, -35]
			if (!itemsCollected[2]) {
				glm::vec3 itemPos3 = glm::vec3(0.0f, 2.0f, -35.0f);
				float distanceToItem3 = glm::length(Pos - itemPos3);
				if (distanceToItem3 < itemCollectionDistance) {
					itemsCollected[2] = true;
					collectedItems++;
					itemSpotlightIntensity[2] = 0.0f; // 开始聚光灯效果
					std::cout << "Collected Rough Lighted Object! Total: " << collectedItems << "/" << totalItems << std::endl;
				}
			}
		}
		
		// 更新道具聚光灯效果
		for(int i = 0; i < 3; i++) {
			if (itemsCollected[i] && itemSpotlightIntensity[i] < 20.0f) {
				itemSpotlightIntensity[i] += spotlightRiseSpeed * deltaT;
				if (itemSpotlightIntensity[i] > 20.0f) {
					itemSpotlightIntensity[i] = 20.0f; // 限制最大强度
				}
			}
		}
		
		if(vel < 0.2) {
			if(currRunState != 1) {
				currRunState = 1;
			}
		} else if(vel < 3.5) {
			if(currRunState != 2) {
				currRunState = 2;
			}
		} else {
			if(currRunState != 3) {
				currRunState = 3;
			}
		}
		
		return deltaT;
	}
};


// This is the main: probably you do not need to touch this!
int main() {
    MISTVALE app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}