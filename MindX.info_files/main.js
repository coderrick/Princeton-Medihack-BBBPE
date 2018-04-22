var camera, scene, renderer, light2, light, pointLight;
var mesh;
var brain = {
	right: null,
	left: null,
	cerebellum: null
};
var options, spawnerOptions, particleSystem, tick,
	clock = new THREE.Clock();

// -----
var mDoParticles = false;
var mParticleCount = 8000; // <-- change this number!
var mParticleSystem;

var mTime = 0.0;
var mTimeStep = (1/60);
var mDuration = 20;

var bufferGeometry;
var prefabGeometry;

var aStartPosition;// = bufferGeometry.createAttribute('aStartPosition', 3);
var aControlPoint1;// = bufferGeometry.createAttribute('aControlPoint1', 3);
var aControlPoint2;// = bufferGeometry.createAttribute('aControlPoint2', 3);
var aEndPosition;// = bufferGeometry.createAttribute('aEndPosition', 3);

var particleCurvePlane = {
	normal: {
		start: { x: [-40, 40], y: [-40, 40], z: [120, 120] },
		middle: { x: [-50, 50], y: [-20, 130], z: [0, 0] },
		end: { x: [-30, 30], y: [10, 40], z: [-90, -90] }
	}
}
function setDefaultParticleCurvePlane(){
	particleCurvePlane.normal.start.x[0] = -40; particleCurvePlane.normal.start.x[1] = 40;
	particleCurvePlane.normal.start.y[0] = -40; particleCurvePlane.normal.start.y[1] = 40;
	particleCurvePlane.normal.start.z[0] = 120; particleCurvePlane.normal.start.z[1] = 120;
	
	particleCurvePlane.normal.middle.x[0] = -50; particleCurvePlane.normal.middle.x[1] = 50;
	particleCurvePlane.normal.middle.y[0] = -20; particleCurvePlane.normal.middle.y[1] = 130;
	particleCurvePlane.normal.middle.z[0] = 0; particleCurvePlane.normal.middle.z[1] = 0;
	
	particleCurvePlane.normal.end.x[0] = -30; particleCurvePlane.normal.end.x[1] = 30;
	particleCurvePlane.normal.end.y[0] = 10; particleCurvePlane.normal.end.y[1] = 40;
	particleCurvePlane.normal.end.z[0] = -90; particleCurvePlane.normal.end.z[1] = -90;
}
// -----
var glitchPass, composer;
// -----

// -----
init();
animate();

function init() {
    camera = new THREE.PerspectiveCamera(70, (window.innerWidth) / (window.innerHeight - 30), 1, 2000000 );
    camera.position.z = -200;
    camera.position.x = -200;

    scene = new THREE.Scene();

    //ambientLight = new THREE.AmbientLight(0x333333);
    light = new THREE.DirectionalLight(0xFFFFFF, 1.0);
    light.position.x = -1000;
    light.position.y = 500;
    light.position.z = -666;
    scene.add(light);
	
	light2 = new THREE.DirectionalLight(0xFFFFFF, 1.0);
    light2.position.x = 1000;
    light2.position.y = 500;
    light2.position.z = 666;
    scene.add(light2);
	
	//var sphere = new THREE.SphereGeometry( 10.5, 16, 8 );
	pointLight = new THREE.PointLight( 0x8899ff, 0, 500 );
	pointLight.position.x = -50;
	pointLight.position.y = 20;
	//pointLight.add( new THREE.Mesh( sphere, new THREE.MeshBasicMaterial( { color: 0xff0040 } ) ) );
	scene.add( pointLight );
    //scene.add(ambientLight);

    var geometry = new THREE.BoxBufferGeometry(200, 200, 200);
    //new THREE.MeshLambertMaterial( { color: 0xdddddd, shading: THREE.SmoothShading } )
    var material = new THREE.MeshLambertMaterial({ color: 0xdddddd, shading: THREE.SmoothShading });
    mesh = new THREE.Mesh(geometry, material);
    //scene.add( mesh );

    renderer = new THREE.WebGLRenderer();
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight - 30);
    // CONTROLS
    cameraControls = new THREE.OrbitControls(camera, renderer.domElement);
    cameraControls.target.set(0, 0, 0);
	
	// Sky
	scene.fog = new THREE.Fog( 0x000000, 1, 5000 );
	scene.fog.color.setHSL( 0.6, 0, 1 );
	var vertexShader = document.getElementById( 'vertexShader' ).textContent;
	var fragmentShader = document.getElementById( 'fragmentShader' ).textContent;
	var uniforms = {
		topColor:    { value: new THREE.Color( 0x001122 ) },
		bottomColor: { value: new THREE.Color( 0x111111 ) },
		offset:      { value: 33 },
		exponent:    { value: 1.6 }
	};
	uniforms.topColor.value.setHSL( 0.6, 1, 0.6 );
	scene.fog.color.copy( uniforms.bottomColor.value );
	var skyGeo = new THREE.SphereGeometry( 4000, 32, 15 );
	var skyMat = new THREE.ShaderMaterial( { vertexShader: vertexShader, fragmentShader: fragmentShader, uniforms: uniforms, side: THREE.BackSide } );
	var sky = new THREE.Mesh( skyGeo, skyMat );
	scene.add( sky );
    // Model
    var loader = new THREE.CTMLoader();
    var textureLoader = new THREE.TextureLoader();
    loader.load("model/brain-left.ctm", function(geometry) {
        var material = new THREE.MeshPhongMaterial({
            specular: 0x303030,
            shininess: 50,
            map: textureLoader.load("model/file1-Left_Hemisphere.jpg")
        });
		geometry.rotateX(0.4);
        callbackModel(geometry, material, 'left');
    }, { useWorker: true });
    loader.load("model/brain-right.ctm", function(geometry) {
        var material = new THREE.MeshPhongMaterial({
            specular: 0x303030,
            shininess: 50,
            map: textureLoader.load("model/file2-Right_Hemisphere.jpg")
        });
		geometry.rotateX(0.4);
        callbackModel(geometry, material, 'right');
    }, { useWorker: true });
    loader.load("model/brain-rest.ctm", function(geometry) {
        var material = new THREE.MeshPhongMaterial({
            specular: 0x303030,
            color: 0xff6633,
            shininess: 50
        });
		geometry.rotateX(0.4);
        callbackModel(geometry, material, 'cerebellum');
    }, { useWorker: true });
    loader.load("model/head.ctm", function(geometry) {
        var material = new THREE.MeshPhongMaterial({
            specular: 0x303030,
            color: 0x6666aa,
            shininess: 50,
            opacity: 0.14,
            transparent: true
        });
		geometry.translate(0, 0, -5);
        callbackModel(geometry, material, 'head');
		
		document.getElementsByClassName('loader')[0].style.display = 'none';
		document.getElementById('navMenu').style.display = 'block';
    }, { useWorker: true });
	
	// Particles
	
	initParticleSystem();

    // Post-processing
	composer = new THREE.EffectComposer( renderer );
	composer.addPass( new THREE.RenderPass( scene, camera ) );
	glitchPass = new THREE.GlitchPass();
	glitchPass.renderToScreen = true;
	composer.addPass( glitchPass );

    //document.body.appendChild( renderer.domElement );
    document.getElementById('scene').appendChild(renderer.domElement);
    window.addEventListener('resize', onWindowResize, false);
	
	onReady();
}

function onReady(){
	document.addEventListener('DOMContentLoaded', function() {
		
		// Disorder DDL
		document.getElementById('ddlDisorder').addEventListener('change', function(event) {
			document.getElementById('currentDisorderHtm').innerHTML = document.querySelector('[disorder-text="' + document.getElementById('ddlDisorder').selectedIndex + '"]').innerHTML;	
			setDefaultParticleCurvePlane();
			switch (document.getElementById('ddlDisorder').selectedIndex){
				case 1:
					particleCurvePlane.normal.middle.y[0] = 90;
					particleCurvePlane.normal.middle.y[1] = 130;
					break;
				case 2:
					particleCurvePlane.normal.middle.y[0] = -20;
					particleCurvePlane.normal.middle.y[1] = 130;
					
					particleCurvePlane.normal.middle.x[0] = -70;
					particleCurvePlane.normal.middle.x[1] = 70;
					
					particleCurvePlane.normal.end.y[0] = 9;
					particleCurvePlane.normal.end.y[1] = 50;
					
					particleCurvePlane.normal.end.z[0] = -80;
					particleCurvePlane.normal.end.z[1] = -72;
					break;
				case 3:
					particleCurvePlane.normal.middle.y[0] = -20;
					particleCurvePlane.normal.middle.y[1] = 130;
					
					particleCurvePlane.normal.middle.x[0] = -70;
					particleCurvePlane.normal.middle.x[1] = 70;
					
					particleCurvePlane.normal.end.y[0] = 5;
					particleCurvePlane.normal.end.y[1] = 38;
					
					particleCurvePlane.normal.end.z[0] = -80;
					particleCurvePlane.normal.end.z[1] = -80;
					break;
			}

			scene.remove( mParticleSystem );
			initParticleSystem();
		}, false);
		document.getElementById('currentDisorderHtm').innerHTML = document.querySelector('[disorder-text="0"]').innerHTML;
		
		// Main DDL
		document.getElementById('ddlMain').addEventListener('change', function(event) {
			document.getElementById('currentMainHtm').innerHTML = document.querySelector('[main-text="' + document.getElementById('ddlMain').selectedIndex + '"]').innerHTML;			
		}, false);
		document.getElementById('currentMainHtm').innerHTML = document.querySelector('[main-text="0"]').innerHTML;
		
		// Control Panel Button
		document.getElementById('btnControlPanelToggle').addEventListener('click', function(event) {
			toggleControlPanel();
		}, false);
		
		// set 
		document.getElementById('ddlDisorder').selectedIndex = 0;
		// styling
		var hPanel = document.querySelector('.control-panel').clientHeight - 40 - 42 - 15 - 15 - 30;
		document.getElementById('currentDisorderHtm').style.height = hPanel + 'px';
		document.getElementById('currentMainHtm').style.height = hPanel + 'px';
		
		// Tabs
		$('a[data-toggle="tab"]').on('shown.bs.tab', function (e) {
			var tab = e.target.getAttribute('aria-controls');
			switch (tab){
				case 'tabMentalDisorders':
					cameraAnimation('detail');
					brain.left.visible = false;
					brain.head.visible = false;
					mDoParticles = true;
					break;
				case 'tabMain':
					cameraAnimation('normal');
					brain.left.visible = true;
					brain.head.visible = true;
					mDoParticles = false;
					break;
			}
		});
	}, false);
	
	if (window.innerWidth < 1200){
		toggleControlPanel();
	} else {
		document.getElementById('btnControlPanelToggle').innerHTML = 'Hide Control Panel';
	}
	if (window.innerWidth <= 640){
		document.querySelector('.control-panel').style.opacity = 1;
		document.querySelector('.control-panel').style.width = 'auto';
		document.querySelector('.control-panel').style.left = '20px';
	}
}

function toggleControlPanel(){
	var panel = document.querySelector('#controlPanel'),
	    style = window.getComputedStyle(panel);
        isHidden = (style.display === 'none');
		
	panel.style.display = (isHidden ? 'block' : 'none');
	
	document.querySelector('.control-panel').style.height = (isHidden ? parseInt(window.innerHeight*0.85) : 56) + 'px';
	var hPanel = document.querySelector('.control-panel').clientHeight - 40 - 42 - 15 - 15 - 30;
	document.getElementById('currentDisorderHtm').style.height = hPanel + 'px';
	document.getElementById('currentMainHtm').style.height = hPanel + 'px';
	document.getElementById('btnControlPanelToggle').innerHTML = (isHidden ? 'Hide Control Panel' : 'Show Control Panel');
}


function cameraAnimation(behavior){
	var coords = {
		detail: {cam: [-121.72, 30.83, -87.36], tg: [-16.77,34.87,-24.07]},
		normal: {cam: [-200, 0, -200], tg: [0, 0, 0]}
	};
	
	new TWEEN.Tween( camera.position ).to( {
		x: coords[behavior].cam[0],
		y: coords[behavior].cam[1],
		z: coords[behavior].cam[2]}, 1600 )
	  .easing( TWEEN.Easing.Sinusoidal.InOut).start();
	
	new TWEEN.Tween( cameraControls.target ).to( {
		x: coords[behavior].tg[0],
		y: coords[behavior].tg[1],
		z: coords[behavior].tg[2]}, 1600 )
	  .easing( TWEEN.Easing.Sinusoidal.InOut).start();
	
	// lights
	if (behavior == 'detail'){
		new TWEEN.Tween( light ).to( {intensity: 0}, 1600 ).easing( TWEEN.Easing.Sinusoidal.InOut).start();
		setTimeout(function(){
			new TWEEN.Tween( pointLight ).to( {intensity: 1.6}, 1600 ).easing( TWEEN.Easing.Sinusoidal.InOut).start();
		}, 1000);	
	} else {
		new TWEEN.Tween( pointLight ).to( {intensity: 0}, 1600 ).easing( TWEEN.Easing.Sinusoidal.InOut).start();
		setTimeout(function(){
			new TWEEN.Tween( light ).to( {intensity: 1}, 1600 ).easing( TWEEN.Easing.Sinusoidal.InOut).start();
		}, 1000);
	}
}

function callbackModel(geometry, material, cache) {
    var mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(0, 20, 0);
    //mesh.scale.set( s, s, s );
    //mesh.rotation.x = rx;
    //mesh.rotation.z = ry;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
	
	//mesh.visible = false;
    scene.add(mesh);
	if (cache){
		brain[cache] = mesh;
	}
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight - 30);
	composer.setSize( window.innerWidth, window.innerHeight );
}

function animate() {
	TWEEN.update();
    requestAnimationFrame(animate);
    //brain.left.rotation.x += 0.005;
    //mesh.rotation.y += 0.01;
	if (mDoParticles) animateParticles();
	mParticleSystem.visible = mDoParticles;
	
    //renderer.render(scene, camera);
	if (mDoParticles && 
		(document.getElementById('ddlDisorder').selectedIndex == 1 || document.getElementById('ddlDisorder').selectedIndex == 5)){
		composer.render();
	} else {
		renderer.render(scene, camera);
	}
	
	cameraControls.update();
}

function animateParticles(){
	mParticleSystem.material.uniforms['uTime'].value = mTime;
	mTime += mTimeStep;
	mTime %= mDuration;
	
	// Schizophrenia
	if (document.getElementById('ddlDisorder').selectedIndex == 1){
		//composer.render();
	}
	// Bipolar
	if (document.getElementById('ddlDisorder').selectedIndex == 2) {
		mParticleSystem.position.x = Math.sin(new Date().getTime() * 0.0025) * 20;	
	}
	// Borderline
	if (document.getElementById('ddlDisorder').selectedIndex == 3) {
		mTime -= 0.1 * Math.sin(new Date().getTime() * 0.00025);
	}
	// Autism
	if (document.getElementById('ddlDisorder').selectedIndex == 4) {
		mDuration -= 10 * Math.sin(new Date().getTime() * 0.025);
	} else {
		mDuration = 20;
	}
}

// *******************************************************************




function initParticleSystem() {
  prefabGeometry = new THREE.CircleGeometry(1.5, 16, 0, Math.PI * 2);
  bufferGeometry = new THREE.BAS.PrefabBufferGeometry(prefabGeometry, mParticleCount);

  bufferGeometry.computeVertexNormals();

  // generate additional geometry data
  var aOffset = bufferGeometry.createAttribute('aOffset', 1);
  aStartPosition = bufferGeometry.createAttribute('aStartPosition', 3);
  aControlPoint1 = bufferGeometry.createAttribute('aControlPoint1', 3);
  aControlPoint2 = bufferGeometry.createAttribute('aControlPoint2', 3);
  aEndPosition = bufferGeometry.createAttribute('aEndPosition', 3);
  var aAxisAngle = bufferGeometry.createAttribute('aAxisAngle', 4);
  var aColor = bufferGeometry.createAttribute('color', 3);

  var i, j, offset;

  // buffer time offset
  var delay;

  for (i = 0, offset = 0; i < mParticleCount; i++) {
    delay = i / mParticleCount * mDuration;

    for (j = 0; j < prefabGeometry.vertices.length; j++) {
      aOffset.array[offset++] = delay;
    }
  }

  // buffer start positions
  var x, y, z;

  for (i = 0, offset = 0; i < mParticleCount; i++) {
    x = 40;
    y = 20;
    z = 10;

    for (j = 0; j < prefabGeometry.vertices.length; j++) {
      aStartPosition.array[offset++] = x;
      aStartPosition.array[offset++] = y;
      aStartPosition.array[offset++] = z;
    }
  }

  // buffer control points

  for (i = 0, offset = 0; i < mParticleCount; i++) {
    x = THREE.Math.randFloat(particleCurvePlane.normal.start.x[0], particleCurvePlane.normal.start.x[1]);
    y = THREE.Math.randFloat(particleCurvePlane.normal.start.y[0], particleCurvePlane.normal.start.y[1]);
    z = THREE.Math.randFloat(particleCurvePlane.normal.start.z[0], particleCurvePlane.normal.start.z[1]);

    for (j = 0; j < prefabGeometry.vertices.length; j++) {
      aControlPoint1.array[offset++] = x;
      aControlPoint1.array[offset++] = y;
      aControlPoint1.array[offset++] = z;
    }
  }

  for (i = 0, offset = 0; i < mParticleCount; i++) {
    x = THREE.Math.randFloat(particleCurvePlane.normal.middle.x[0], particleCurvePlane.normal.middle.x[1]);
    y = THREE.Math.randFloat(particleCurvePlane.normal.middle.y[0], particleCurvePlane.normal.middle.y[1]);
    z = THREE.Math.randFloat(particleCurvePlane.normal.middle.z[0], particleCurvePlane.normal.middle.z[1]);

    for (j = 0; j < prefabGeometry.vertices.length; j++) {
      aControlPoint2.array[offset++] = x;
      aControlPoint2.array[offset++] = y;
      aControlPoint2.array[offset++] = z;
    }
  }

  // buffer end positions

  for (i = 0, offset = 0; i < mParticleCount; i++) {
    x = THREE.Math.randFloat(particleCurvePlane.normal.end.x[0], particleCurvePlane.normal.end.x[1]); //10;
    y = THREE.Math.randFloat(particleCurvePlane.normal.end.y[0], particleCurvePlane.normal.end.y[1]); //35;
    z = THREE.Math.randFloat(particleCurvePlane.normal.end.z[0], particleCurvePlane.normal.end.z[1]);//-90;

    for (j = 0; j < prefabGeometry.vertices.length; j++) {
      aEndPosition.array[offset++] = x;
      aEndPosition.array[offset++] = y;
      aEndPosition.array[offset++] = z;
    }
  }

  // buffer axis angle
  var axis = new THREE.Vector3();
  var angle = 0;

  for (i = 0, offset = 0; i < mParticleCount; i++) {
    axis.x = THREE.Math.randFloatSpread(18);
    axis.y = THREE.Math.randFloatSpread(18);
    axis.z = THREE.Math.randFloatSpread(18);
    axis.normalize();

    angle = Math.PI * THREE.Math.randInt(16, 18);

    for (j = 0; j < prefabGeometry.vertices.length; j++) {
      aAxisAngle.array[offset++] = axis.x;
      aAxisAngle.array[offset++] = axis.y;
      aAxisAngle.array[offset++] = axis.z;
      aAxisAngle.array[offset++] = angle;
    }
  }

  // buffer color
  var color = new THREE.Color();
  var h, s, l;

  for (i = 0, offset = 0; i < mParticleCount; i++) {
    h = i / mParticleCount;
    s = THREE.Math.randFloat(0.2, 0.6);
    l = THREE.Math.randFloat(0.1, 0.6);

    color.setHSL(h, s, l);

    for (j = 0; j < prefabGeometry.vertices.length; j++) {
      aColor.array[offset++] = color.r;
      aColor.array[offset++] = color.g;
      aColor.array[offset++] = color.b;
    }
  }


  var material = new THREE.BAS.PhongAnimationMaterial(
    // custom parameters & THREE.MeshPhongMaterial parameters
    {
      vertexColors: THREE.VertexColors,
      shading: THREE.FlatShading,
      side: THREE.DoubleSide,
      uniforms: {
        uTime: {type: 'f', value: 0},
        uDuration: {type: 'f', value: mDuration}
      },
      shaderFunctions: [
        THREE.BAS.ShaderChunk['quaternion_rotation'],
        THREE.BAS.ShaderChunk['cubic_bezier']
      ],
      shaderParameters: [
        'uniform float uTime;',
        'uniform float uDuration;',
        'attribute float aOffset;',
        'attribute vec3 aStartPosition;',
        'attribute vec3 aControlPoint1;',
        'attribute vec3 aControlPoint2;',
        'attribute vec3 aEndPosition;',
        'attribute vec4 aAxisAngle;'
      ],
      shaderVertexInit: [
        'float tProgress = mod((uTime + aOffset), uDuration) / uDuration;',

        'float angle = aAxisAngle.w * tProgress;',
        'vec4 tQuat = quatFromAxisAngle(aAxisAngle.xyz, angle);'
      ],
      shaderTransformNormal: [
        'objectNormal = rotateVector(tQuat, objectNormal);'
      ],
      shaderTransformPosition: [
        'transformed = rotateVector(tQuat, transformed);',
        'transformed += cubicBezier(aStartPosition, aControlPoint1, aControlPoint2, aEndPosition, tProgress);'
      ]
    },
    // THREE.MeshPhongMaterial uniforms
    {
      specular: 0xff0000,
      shininess: 20
    }
  );

  mParticleSystem = new THREE.Mesh(bufferGeometry, material);
  // because the bounding box of the particle system does not reflect its on-screen size
  // set this to false to prevent the whole thing from disappearing on certain angles
  mParticleSystem.frustumCulled = false;
	
  scene.add(mParticleSystem);
}

