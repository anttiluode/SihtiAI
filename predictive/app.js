/* Offline UI. Input capture never leaves the local browser. */
(function(){'use strict';
  const $=id=>document.getElementById(id);
  const W=48,H=36,Head=globalThis.ResidueHead,World=globalThis.ResidueWorld;
  const head=new Head({width:W,height:H});
  const view={current:$('current'),wake:$('wake'),prediction:$('prediction'),surprise:$('surprise')};
  const source=$('source'),file=$('file'),video=$('video');
  const sampleCtx=$('sampler').getContext('2d',{willReadFrequently:true});
  let timer=null,syntheticT=0,direction=1,cameraStream=null,videoUrl=null,sourceReady=false;
  let changeAt=-1,chartData=[];
  const setStatus=s=>{$('status').textContent=s;};
  function stop(){if(timer!==null){clearInterval(timer);timer=null;}$('run').textContent='Start';}
  function release(){stop();if(cameraStream){for(const track of cameraStream.getTracks())track.stop();cameraStream=null;}
    video.pause();video.srcObject=null;
    if(videoUrl){URL.revokeObjectURL(videoUrl);videoUrl=null;}
    video.removeAttribute('src');sourceReady=false;
  }
  function paint(name,array){const ctx=view[name].getContext('2d');const img=ctx.createImageData(W,H);
    for(let i=0;i<W*H;i++){
      const j=i*4;let r,g,b;const v=array?array[i]:0;
      if(name==='wake'){
        const z=Math.min(1,Math.abs(v)*7);
        r=v<0?Math.round(36+206*z):20;
        g=v>0?Math.round(36+205*z):20;
        b=Math.round(26+74*z);
      }else if(name==='surprise'){
        const z=Math.min(1,Math.abs(v)*8);
        r=Math.round(9+244*z);g=Math.round(14+180*z);b=Math.round(22+36*z);
      }else{let z=Math.min(1,Math.max(0,v));r=g=b=Math.round(z*255);}
      img.data[j]=r;img.data[j+1]=g;img.data[j+2]=b;img.data[j+3]=255;
    }
    ctx.putImageData(img,0,0);
  }
  function frameFromVideo(){
    if(video.readyState<2||video.videoWidth<1)return null;
    if(video.ended){stop();setStatus('Video ended. The stored bridges remain; play again to test recall.');return null;}
    sampleCtx.drawImage(video,0,0,W,H);
    const rgb=sampleCtx.getImageData(0,0,W,H).data,gray=new Float32Array(W*H);
    for(let i=0;i<W*H;i++){const p=i*4;gray[i]=(.2126*rgb[p]+.7152*rgb[p+1]+.0722*rgb[p+2])/255;}
    return gray;
  }
  function tick(){
    let frame;
    if(source.value==='file'||source.value==='camera')frame=frameFromVideo();
    else {frame=World.render(source.value,syntheticT,W,H);syntheticT+=direction;}
    if(!frame)return;
    const result=head.push(frame);
    paint('current',frame);paint('wake',result.forecast.wake);
    paint('prediction',result.forecast.predicted);
    if(result.completed)paint('surprise',result.completed.error);
    $('frames').textContent=result.frames;
    $('bridges').textContent=result.bankSize;
    const score=result.completed;
    if(score){chartData.push({m:score.modelError,b:score.holdError});if(chartData.length>220)chartData.shift();drawChart();}
    const recent=chartData.slice(-80);
    if(recent.length){
      const m=recent.reduce((s,d)=>s+d.m,0)/recent.length;
      const b=recent.reduce((s,d)=>s+d.b,0)/recent.length;
      $('modelError').textContent=(m*1000).toFixed(3);
      $('holdError').textContent=(b*1000).toFixed(3);
      const delta=b>1e-12?(1-m/b)*100:0;
      $('advantage').textContent=(delta>=0?'+':'')+delta.toFixed(1)+'%';
      $('advantage').style.color=delta>=0?'var(--mint)':'var(--rose)';
    }
    const f=result.forecast;
    $('confidence').textContent='Readout confidence: '+(f.confidence*100).toFixed(1)+'% · key distance: '+(f.distance==null?'—':f.distance.toFixed(5));
    $('topHits').textContent=f.hits.length?f.hits.slice(0,4).map((h,i)=>
      '#'+(i+1)+' · age '+h.age+' frames · contribution '+(100*h.weight).toFixed(1)+'%').join('  |  '):'No old transitions to retrieve.';
    if(result.frames===1)setStatus('First state observed. There is no completed transition yet.');
    else if(result.frames===20)setStatus('Learning from completed transitions. Let a full loop repeat; watch the two error curves.');
    else if(result.frames===110 && source.value==='bounce')setStatus('The pulse has returned to states seen earlier. This is where associative reactivation can begin to help.');
  }
  function drawChart(){const canvas=$('chart'),ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
    ctx.clearRect(0,0,w,h);ctx.fillStyle='#09161d';ctx.fillRect(0,0,w,h);
    if(chartData.length<2)return;
    const max=Math.max(1e-7,...chartData.flatMap(d=>[d.m,d.b]))*1.05;
    ctx.strokeStyle='#29424d';ctx.lineWidth=1;
    for(let i=1;i<=4;i++){const y=(i/5)*h;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}
    for(const [key,color] of [['b','#ffd58a'],['m','#80ebce']]){
      ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();
      chartData.forEach((d,i)=>{const x=i/(chartData.length-1)*(w-14)+7,y=h-12-d[key]/max*(h-25);
        if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();
    }
    ctx.fillStyle='#9ab0b8';ctx.font='12px sans-serif';ctx.fillText('max '+(max*1000).toFixed(2)+' ×10⁻³ MSE',12,16);
  }
  function erase(resetWorld=false){head.reset();chartData=[];drawChart();
    if(resetWorld){syntheticT=0;direction=1;}
    for(const k of Object.keys(view))paint(k,null);
    for(const id of ['frames','bridges'])$(id).textContent='0';
    for(const id of ['modelError','holdError','advantage'])$(id).textContent='—';
    $('topHits').textContent='No completed memories yet.';$('confidence').textContent='Readout confidence: —';
    setStatus('Memory cleared. No previously stored transitions remain.');
  }
  async function start(){
    if(timer!==null){stop();setStatus('Paused. The memory and recorded errors are preserved.');return;}
    if(source.value==='file'&&!videoUrl){file.click();return;}
    if(source.value==='camera'&&!cameraStream){
      if(!navigator.mediaDevices?.getUserMedia){setStatus('Webcam unavailable in this browser/context. Try video file or the built-in motion worlds.');return;}
      try{cameraStream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user'},audio:false});
        video.srcObject=cameraStream;video.muted=true;await video.play();}
      catch(e){setStatus('Camera could not start: '+e.message+'. Video files and synthetic worlds still work.');return;}
    }
    if(source.value==='file'){
      try{if(video.ended)video.currentTime=0;await video.play();}
      catch(e){setStatus('Video could not play: '+e.message);return;}
    }
    timer=setInterval(tick,1000/Number($('rate').value));
    $('run').textContent='Pause';setStatus('Streaming. Each prediction is made before the next observation.');
  }
  $('run').addEventListener('click',start);
  $('step').addEventListener('click',()=>{if(timer!==null)stop();if(['file','camera'].includes(source.value)){setStatus('For video/camera press Start. Step once works on built-in sources.');return;}tick();});
  $('reset').addEventListener('click',()=>erase(true));
  $('reverse').addEventListener('click',()=>{
    if(['file','camera'].includes(source.value)){setStatus('Reverse applies to built-in simulated motion only.');return;}
    direction*=-1;setStatus('Intervention: motion reversed without erasing memory. Watch surprise and subsequent readout.');
  });
  $('address').addEventListener('change',()=>{head.mode=$('address').value;
    setStatus(head.mode==='gist'?'Address now ignores movement history. Opposite motion at the same location can become ambiguous.':'Address now includes the last movement as well as current shape. Memory was not erased.');});
  $('halflife').addEventListener('change',()=>{head.halfLife=Number($('halflife').value);setStatus('Changed how quickly old transitions lose influence. Memory entries were preserved.');});
  $('correction').addEventListener('change',()=>{head.feedback=$('correction').checked?0.06:0;
    setStatus(head.feedback?'Surprise now corrects nearby old transitions and writes new bridges.':'Correction of old transitions disabled; new completed transitions are still stored.');});
  $('learning').addEventListener('change',()=>{head.learning=$('learning').checked;
    setStatus(head.learning?'Learning resumed from newly observed transitions.':'Learning frozen. Testing only using already stored transitions.');});
  $('rate').addEventListener('change',()=>{if(timer!==null){clearInterval(timer);timer=setInterval(tick,1000/Number($('rate').value));}});
  source.addEventListener('change',()=>{
    const choice=source.value;release();
    if(choice==='file'){file.click();setStatus('Choose a video from your computer. Only small grayscale frames are processed; nothing is uploaded.');}
    else if(choice==='camera')setStatus('Press Start to request webcam permission. For file:// restrictions, use a video file or built-in world.');
    else setStatus('Changed the signal to '+choice+'. Existing memory remains, so you can observe how it handles an unfamiliar world.');
  });
  file.addEventListener('change',()=>{
    if(!file.files?.length)return;
    release();videoUrl=URL.createObjectURL(file.files[0]);video.src=videoUrl;video.loop=false;video.muted=true;
    setStatus('Selected '+file.files[0].name+'. Click Start to predict incoming video frames locally.');
    source.value='file';
  });
  $('save').addEventListener('click',()=>{
    const lines=['t,model_mse,hold_last_mse,retrieval_confidence,stored_bridges'];
    for(const row of head.trials)lines.push([row.t,row.modelError,row.holdError,row.confidence,row.bankSize].join(','));
    const blob=new Blob([lines.join('\n')+'\n'],{type:'text/csv'}),url=URL.createObjectURL(blob),a=document.createElement('a');
    a.href=url;a.download='residue_head_receipt.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    setStatus('Saved local prediction-error history as a CSV. No camera pixels were exported.');
  });
  for(const k of Object.keys(view))paint(k,null);
  drawChart();
  window.addEventListener('beforeunload',release);
})();

