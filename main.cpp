#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		
#include <GL/freeglut.h>	
#endif

const unsigned int windowWidth = 600, windowHeight = 600;
unsigned int shaderProgram;

struct vec2 {
	
	float x, y;
	vec2(float _x = 0, float _y = 0) { x = _x; y = _y; }
};

struct vec3 {
	float x, y, z;
	vec3(float _x = 0, float _y = 0, float _z = 0) { x = _x; y = _y; z = _z; }
	vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }
	vec3 operator+(const vec3& v) const { return vec3(x + v.x, y + v.y, z + v.z); }
	vec3 operator-(const vec3& v) const { return vec3(x - v.x, y - v.y, z - v.z); }
	vec3 operator*(const vec3& v) const { return vec3(x * v.x, y * v.y, z * v.z); }
	vec3 operator-() const { return vec3(-x, -y, -z); }
	vec3 normalize() const { return (*this) * (1.0f / (Length() + 0.000001)); }
	float Length() const { return sqrtf(x * x + y * y + z * z); }

	void SetUniform(char * name) {
		int location = glGetUniformLocation(shaderProgram, name);
		if (location >= 0) glUniform3fv(location, 1, &x);
		else printf("uniform %s cannot be set\n", name);
	}
};

float dot(const vec3& v1, const vec3& v2) { return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z); }

vec3 cross(const vec3& v1, const vec3& v2) { return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x); }

struct mat4 { 
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}

	void SetUniform(char * name) {
		int location = glGetUniformLocation(shaderProgram, name);
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, &m[0][0]);
		else printf("uniform %s cannot be set\n", name);
	}
};

mat4 TranslateMatrix(vec3 t) {
	return mat4(1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		t.x, t.y, t.z, 1);
}

mat4 ScaleMatrix(vec3 s) {
	return mat4(s.x, 0, 0, 0,
		0, s.y, 0, 0,
		0, 0, s.z, 0,
		0, 0, 0, 1);
}

mat4 RotationMatrix(float angle, vec3 w) {
	float c = cosf(angle), s = sinf(angle);
	w = w.normalize();
	return mat4(c * (1 - w.x*w.x) + w.x*w.x, w.x*w.y*(1 - c) + w.z*s, w.x*w.z*(1 - c) - w.y*s, 0,
		w.x*w.y*(1 - c) - w.z*s, c * (1 - w.y*w.y) + w.y*w.y, w.y*w.z*(1 - c) + w.x*s, 0,
		w.x*w.z*(1 - c) + w.y*s, w.y*w.z*(1 - c) - w.x*s, c * (1 - w.z*w.z) + w.z*w.z, 0,
		0, 0, 0, 1);
}

struct Camera { 
	vec3 wEye, wLookat, wVup;
	float fov, asp, fp, bp;
public:
	Camera() {
		asp = 1;
		fov = 80.0f * (float)M_PI / 180.0f;
		fp = 0.1; bp = 100;
	}
	mat4 V() { 
		vec3 w = (wEye - wLookat).normalize();
		vec3 u = cross(wVup, w).normalize();
		vec3 v = cross(w, u);
		return TranslateMatrix(-wEye) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}
	mat4 P() { 
		return mat4(1 / (tan(fov / 2)*asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp*bp / (bp - fp), 0);
	}

	void SetUniform() {
		int location = glGetUniformLocation(shaderProgram, "wEye");
		if (location >= 0) glUniform3fv(location, 1, &wEye.x);
		else printf("uniform wEye cannot be set\n");
	}
};

Camera camera; 

struct Material {
	vec3 kd, ks, ka;
	float shininess;

	void SetUniform() {
		kd.SetUniform("kd");
		ks.SetUniform("ks");
		ka.SetUniform("ka");
		int location = glGetUniformLocation(shaderProgram, "shine");
		if (location >= 0) glUniform1f(location, shininess); else printf("uniform shininess cannot be set\n");
	}
};

struct Light {
	vec3 La, Le;
	vec3 wLightDir;

	Light() : La(1, 1, 1), Le(3, 3, 3) { }
	void SetUniform(bool enable) {
		if (enable) {
			La.SetUniform("La");
			Le.SetUniform("Le");
		}
		else {
			vec3(0,0,0).SetUniform("La");
			vec3(0, 0, 0).SetUniform("Le");
		}
		wLightDir.SetUniform("wLiDir");
	}
};

class Shader {
	void getErrorInfo(unsigned int handle) {
		int logLen, written;
		glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
		if (logLen > 0) {
			char * log = new char[logLen];
			glGetShaderInfoLog(handle, logLen, &written, log);
			printf("Shader log:\n%s", log);
			delete log;
		}
	}
	void checkShader(unsigned int shader, char * message) { 	
		int OK;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
		if (!OK) { printf("%s!\n", message); getErrorInfo(shader); getchar(); }
	}
	void checkLinking(unsigned int program) { 	
		int OK;
		glGetProgramiv(program, GL_LINK_STATUS, &OK);
		if (!OK) { printf("Failed to link shader program!\n"); getErrorInfo(program); getchar(); }
	}

public:
	void Create(const char * vertexSource, const char * fragmentSource, const char * fsOuputName) {
		
		unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
		if (!vertexShader) { printf("Error in vertex shader creation\n"); exit(1); }
		glShaderSource(vertexShader, 1, &vertexSource, NULL);
		glCompileShader(vertexShader);
		checkShader(vertexShader, "Vertex shader error");

		
		unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		if (!fragmentShader) { printf("Error in fragment shader creation\n"); exit(1); }
		glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
		glCompileShader(fragmentShader);
		checkShader(fragmentShader, "Fragment shader error");

		
		shaderProgram = glCreateProgram();
		if (!shaderProgram) { printf("Error in shader program creation\n"); exit(1); }
		glAttachShader(shaderProgram, vertexShader);
		glAttachShader(shaderProgram, fragmentShader);

		
		glBindFragDataLocation(shaderProgram, 0, fsOuputName);	

		
		glLinkProgram(shaderProgram);
		checkLinking(shaderProgram);
	}
	virtual void Bind(mat4 M, mat4 Minv) = 0;
	~Shader() { glDeleteProgram(shaderProgram); }
};

class PhongShader : public Shader {
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		uniform mat4  MVP, M, Minv; 
		uniform vec3  wLiDir;       
		uniform vec3  wEye;         

		layout(location = 0) in vec3  vtxPos;            
		layout(location = 1) in vec3  vtxNorm;      	 
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    
		out vec3 wView;             
		out vec3 wLight;		    

		void main() {
		   gl_Position = vec4(vtxPos, 1) * MVP; 
		   vec4 wPos = vec4(vtxPos, 1) * M;
		   wLight  = wLiDir;
		   wView   = wEye - wPos.xyz;
		   wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		}
	)";

	
	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		uniform vec3 kd, ks, ka; 
		uniform vec3 La, Le;     
		uniform float shine;     

		in  vec3 wNormal;       
		in  vec3 wView;         
		in  vec3 wLight;        
		in vec2 texcoord;
		out vec4 fragmentColor; 

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			vec3 L = normalize(wLight);
			vec3 H = normalize(L + V);
			float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
			vec3 color = ka * La + (kd * cost + ks * pow(cosd,shine)) * Le;
			fragmentColor = vec4(color, 1);
		}
	)";
public:
	PhongShader() { Create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(mat4 M, mat4 Minv) {
		glUseProgram(shaderProgram); 		
		mat4 MVP = M * camera.V() * camera.P();
		MVP.SetUniform("MVP");
		M.SetUniform("M");
		Minv.SetUniform("Minv");
	}
};

Shader * shader = NULL;

struct VertexData {
	vec3 position, normal;
	vec2 texcoord;
};

class Geometry {
	unsigned int vao, type;        
protected:
	int nVertices;
public:
	Geometry(unsigned int _type) {
		type = _type;
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
	}
	void Draw(mat4 M, mat4 Minv) {
		shader->Bind(M, Minv);
		glBindVertexArray(vao);
		glDrawArrays(type, 0, nVertices);
	}
};

class ParamSurface : public Geometry {
public:
	ParamSurface() : Geometry(GL_TRIANGLES) {}

	virtual VertexData GenVertexData(float u, float v) = 0;

	void Create(int N = 16, int M = 16) {
		unsigned int vbo;
		glGenBuffers(1, &vbo); 
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		nVertices = N * M * 6;
		std::vector<VertexData> vtxData;	
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < M; j++) {
				vtxData.push_back(GenVertexData((float)i / N, (float)j / M));
				vtxData.push_back(GenVertexData((float)(i + 1) / N, (float)j / M));
				vtxData.push_back(GenVertexData((float)i / N, (float)(j + 1) / M));
				vtxData.push_back(GenVertexData((float)(i + 1) / N, (float)j / M));
				vtxData.push_back(GenVertexData((float)(i + 1) / N, (float)(j + 1) / M));
				vtxData.push_back(GenVertexData((float)i / N, (float)(j + 1) / M));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVertices * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		
		glEnableVertexAttribArray(0);  
		glEnableVertexAttribArray(1);  
		glEnableVertexAttribArray(2);  
		
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}
};

class Sphere : public ParamSurface {
	float r;
public:
	Sphere(float _r) { 
		r = _r;
		Create(20, 20);
	}

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.normal = vec3(cosf(u * 2.0f * M_PI) * sin(v*M_PI), sinf(u * 2.0f * M_PI) * sinf(v*M_PI), cosf(v*M_PI));
		vd.position = vd.normal * r;
		vd.texcoord = vec2(u, v);
		return vd;
	}
};

class TruncatedCone : public ParamSurface {
	float rStart, rEnd;
public:
	TruncatedCone(float _rStart, float _rEnd) { 
		rStart = _rStart, rEnd = _rEnd; 
		Create(20, 20);
	}

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		float U = u * 2.0f * M_PI;		
		vec3 circle = vec3(cosf(U), sinf(U), 0);
		vd.position = circle * (rStart * (1 - v) + rEnd * v) + vec3(0, 0, v);
		vec3 drdU = vec3(-sinf(U), cosf(U));
		vec3 drdv = circle * (rEnd - rStart) + vec3(0, 0, 1);
		vd.normal = cross(drdU, drdv);
		vd.texcoord = vec2(u, v);
		return vd;
	}
};

class Quad : public ParamSurface {
	float size;
public:
	Quad() {
		size = 100;
		Create(20, 20);
	}

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.normal = vec3(0, 1, 0);
		vd.position = vec3((u - 0.5) * 2, 0, (v - 0.5) * 2) * size;
		vd.texcoord = vec2(u, v);
		return vd;
	}
};

class Floor {
	Material * material;
	Geometry * quad;
public:
	Floor(Material * _m) {
		material = _m;
		quad = new Quad();
	}
	void Draw(mat4 M, mat4 Minv) {
		material->SetUniform();
		quad->Draw(M, Minv);
	}
};

const float boneRadius = 0.5;
const float legLength = 5;

#define INVERSE_KINEMATICS
class PrimitiveMan {
	Material * material;
	Sphere * head;
	TruncatedCone * torso;
	Sphere * joint;
	TruncatedCone * bone;

	float dleftarm_angle, drightarm_angle, dleftleg_angle, drightleg_angle;
	float leftLegAngle, rightLegAngle, leftArmAngle, rightArmAngle, leftToeAngle, rightToeAngle;
	float forward, up;          
public:
	PrimitiveMan(Material * _m) {
		material = _m;
		head = new Sphere(1.5);
		torso = new TruncatedCone(1.0, 0.8);
		joint = new Sphere(boneRadius);
		bone = new TruncatedCone(boneRadius, boneRadius/5);
		forward = 0;
		up = legLength + boneRadius;

		dleftarm_angle = -6; drightarm_angle = 6;
		dleftleg_angle = 3;  drightleg_angle = -3;

		rightLegAngle = 120;
		rightToeAngle = -120;
		leftLegAngle = 60;
		leftToeAngle = -60;
		rightArmAngle = 30;
		leftArmAngle = 150;
	}
	float Forward() { return forward; }

	void Animate(float dt) {
		if (forward < 105) {
			float oldleg_angle = rightLegAngle;

			leftArmAngle += dleftarm_angle * dt;
			rightArmAngle += drightarm_angle * dt;
			leftLegAngle += dleftleg_angle * dt;
			rightLegAngle += drightleg_angle * dt;
			if (leftArmAngle  > 150) { dleftarm_angle = -6; drightarm_angle = 6; }
			if (rightArmAngle > 150) { dleftarm_angle = 6; drightarm_angle = -6; }
			if (leftLegAngle  > 120) { dleftleg_angle = -3; drightleg_angle = 3; }
			if (rightLegAngle > 120) { dleftleg_angle = 3; drightleg_angle = -3; }
			
#ifdef INVERSE_KINEMATICS
			forward += fabs(legLength * (sin((rightLegAngle - 90) * M_PI / 180) -sin((oldleg_angle - 90) * M_PI / 180)));
			up = legLength * cos((rightLegAngle - 90) * M_PI / 180) + boneRadius;
			leftToeAngle = -leftLegAngle;
			rightToeAngle = -rightLegAngle;
#else
			forward += 0.3 * dt;
#endif
		}
		else {
			up -= 2 * dt;
		}
	}
	void DrawHead(mat4 M, mat4 Minv) {
		M = TranslateMatrix(vec3(0, 6.5f, 0)) * M;
		Minv = Minv * TranslateMatrix(-vec3(0, 6.5f, 0));
		head->Draw(M, Minv);
	}
	void DrawTorso(mat4 M, mat4 Minv) {
		M = ScaleMatrix(vec3(2, 1, 5)) * RotationMatrix(90 * M_PI / 180, vec3(1, 0, 0)) * TranslateMatrix(vec3(0, 5, 0)) * M;
		Minv = Minv * TranslateMatrix(-vec3(0, 5, 0)) * RotationMatrix(-90 * M_PI / 180, vec3(1, 0, 0)) * ScaleMatrix(vec3(0.5, 1, 0.2));
		torso->Draw(M, Minv);
	}
	void DrawLeftLeg(mat4 M, mat4 Minv) {
		joint->Draw(M, Minv);

		M = RotationMatrix(leftLegAngle * M_PI / 180, vec3(1, 0, 0)) * M;
		Minv = Minv * RotationMatrix(-leftLegAngle * M_PI / 180, vec3(1, 0, 0));
		bone->Draw(ScaleMatrix(vec3(1, 1, legLength)) * TranslateMatrix(vec3(0, 0, boneRadius)) * M,
			       Minv * TranslateMatrix(-vec3(0, 0, boneRadius)) * ScaleMatrix(vec3(1, 1, 1/legLength)));

		DrawToe(RotationMatrix(-leftLegAngle * M_PI / 180, vec3(1, 0, 0)) * TranslateMatrix(vec3(0, 0, legLength)) * M,
			Minv * TranslateMatrix(-vec3(0, 0, legLength)) * RotationMatrix(leftLegAngle * M_PI / 180, vec3(1, 0, 0)));
	}

	void DrawRightLeg(mat4 M, mat4 Minv) {
		joint->Draw(M, Minv);

		M = RotationMatrix(rightLegAngle * M_PI / 180, vec3(1, 0, 0)) * M;
		Minv = Minv * RotationMatrix(-rightLegAngle * M_PI / 180, vec3(1, 0, 0));
		const float legLength = 5;
		bone->Draw(ScaleMatrix(vec3(1, 1, legLength)) * TranslateMatrix(vec3(0, 0, boneRadius)) * M,
			Minv * TranslateMatrix(-vec3(0, 0, boneRadius)) * ScaleMatrix(vec3(1, 1, 1 / legLength)));

		DrawToe(RotationMatrix(-rightLegAngle * M_PI / 180, vec3(1, 0, 0)) * TranslateMatrix(vec3(0, 0, legLength)) *M,
			Minv  * TranslateMatrix(-vec3(0, 0, legLength))* RotationMatrix(rightLegAngle * M_PI / 180, vec3(1, 0, 0)));
	}

	void DrawToe(mat4 M, mat4 Minv) {
		joint->Draw(M, Minv);
		const float toeLength = 1;
		bone->Draw(ScaleMatrix(vec3(1, 1, toeLength)) * TranslateMatrix(vec3(0, 0, boneRadius)) * M,
			       Minv * TranslateMatrix(-vec3(0, 0, boneRadius)) * ScaleMatrix(vec3(1, 1, 1 / toeLength)));
	}

	void DrawArm(mat4 M, mat4 Minv) {
		joint->Draw(M, Minv);
		const float toeLength = 1;
		bone->Draw(ScaleMatrix(vec3(1, 1, 4)) * TranslateMatrix(vec3(0, 0, boneRadius)) * M,
			Minv * TranslateMatrix(-vec3(0, 0, boneRadius)) * ScaleMatrix(vec3(1, 1, 1.0 / 4)));
	}

	void Draw(mat4 M, mat4 Minv) {     
		M = TranslateMatrix(vec3(0, up, forward)) * M;
		Minv = Minv * TranslateMatrix(-vec3(0, up, forward));
		material->SetUniform();
		DrawHead(M, Minv);
		DrawTorso(M, Minv);

		vec3 rightLegJoint(-2, 0, 0);
		DrawRightLeg(TranslateMatrix(rightLegJoint) * M, Minv * TranslateMatrix(-rightLegJoint));

		vec3 leftLegJoint( 2, 0, 0);
		DrawLeftLeg(TranslateMatrix(leftLegJoint) * M, Minv * TranslateMatrix(-leftLegJoint));

		vec3 rightArmJoint(-2.4, 5, 0);
		DrawArm(RotationMatrix(rightArmAngle * M_PI / 180, vec3(1, 0, 0)) * TranslateMatrix(rightArmJoint) * M,
			Minv * TranslateMatrix(-rightArmJoint) * RotationMatrix(-rightArmAngle * M_PI / 180, vec3(1, 0, 0)));

		vec3 leftArmJoint(2.4, 5, 0);
		DrawArm(RotationMatrix(leftArmAngle * M_PI / 180, vec3(1, 0, 0)) * TranslateMatrix(leftArmJoint) * M,
			Minv * TranslateMatrix(-leftArmJoint) * RotationMatrix(-leftArmAngle * M_PI / 180, vec3(1, 0, 0)));
	}
};

class Scene {
	PrimitiveMan * pman;
	Floor * floor;
public:
	Light light;

	void Build() {
		
		shader = new PhongShader();

		
		Material * material0 = new Material;
		material0->kd = vec3(0.2f, 0.3f, 1);
		material0->ks = vec3(1, 1, 1);
		material0->ka = vec3(0.2f, 0.3f, 1);
		material0->shininess = 20;

		Material * material1 = new Material;
		material1->kd = vec3(0, 1, 1);
		material1->ks = vec3(2, 2, 2);
		material1->ka = vec3(0.2f, 0.2f, 0.2f);
		material1->shininess = 200;

		
		pman = new PrimitiveMan(material0);
		floor = new Floor(material1);

		
		camera.wEye = vec3(0, 0, 4);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 1, 0);

		
		light.wLightDir = vec3(5, 5, 4);

	}
	void Render() {
		camera.SetUniform();
		light.SetUniform(true);

		mat4 unit = TranslateMatrix(vec3(0, 0, 0));
		floor->Draw(unit, unit);

		pman->Draw(unit, unit);
		

		light.SetUniform(false);

		mat4 shadowMatrix = { 1, 0, 0, 0,
			-light.wLightDir.x / light.wLightDir.y, 0, -light.wLightDir.z / light.wLightDir.y, 0,
			0, 0, 1, 0,
			0, 0.001f, 0, 1 };
		pman->Draw(shadowMatrix, shadowMatrix);
	}

	void Animate(float t) {
		static float tprev = 0;
		float dt = t - tprev;
		tprev = t;

		pman->Animate(dt);

		static float cam_angle = 0;
		cam_angle += 0.01 * dt;			

		const float camera_rad = 30;
		camera.wEye = vec3(cos(cam_angle) * camera_rad, 10, sin(cam_angle) * camera_rad + pman->Forward());
		camera.wLookat = vec3(0, 0, pman->Forward());
	}
};

Scene scene;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
	scene.Render();
	glutSwapBuffers();									
}

void onKeyboard(unsigned char key, int pX, int pY) { }

void onKeyboardUp(unsigned char key, int pX, int pY) { }

void onMouse(int button, int state, int pX, int pY) { }

void onMouseMotion(int pX, int pY) {
}

void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); 
	float sec = time / 100.0f;				
	scene.Animate(sec);					
	glutPostRedisplay();					
}

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(3, 3);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				
	glutInitWindowPosition(100, 100);							
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	
	glewInit();
#endif
	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	int majorVersion, minorVersion;
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	return 1;
}
