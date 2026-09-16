// Render only local fixtures in a headless Chromium process; no live cameras.
const fs=require('node:fs'),path=require('node:path'),http=require('node:http');
const {chromium}=require('playwright');
const root=path.resolve(__dirname,'../..'),rig=process.env.SIM_RIG||'side',output=path.join(root,'review/simulation',rig==='front'?'front':'');
const count=Number(process.argv[2]||480),fps=15;
const mime={'.html':'text/html','.js':'text/javascript','.glb':'model/gltf-binary'};
const server=http.createServer((req,res)=>{
 const target=path.resolve(root,'.'+decodeURIComponent(new URL(req.url,'http://localhost').pathname));
 if(!target.startsWith(root+path.sep)){res.writeHead(403).end();return;}
 fs.readFile(target,(error,data)=>{if(error){res.writeHead(404).end();return;}res.setHeader('Content-Type',mime[path.extname(target)]||'application/octet-stream');res.end(data);});
});
(async()=>{
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 let browser;
 try{
  browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
  const page=await browser.newPage();page.on('pageerror',e=>console.error(e));
  await page.goto(`http://127.0.0.1:${server.address().port}/scripts/simulation/scene.html?rig=${rig}`);
  await page.waitForFunction(()=>window.ready,{},{timeout:60000});
  const info=await page.evaluate(()=>window.info);console.log(JSON.stringify(info));
  fs.mkdirSync(path.join(output,'frames'),{recursive:true});
  const manifest={...info,rig,fps,frames:[]};
  const base=rig==='front'?JSON.parse(fs.readFileSync(path.join(root,'review/simulation/manifest.json'))):null;
  for(let frame=0;frame<count;frame++)for(let camera=0;camera<3;camera++){
   if(base&&camera<2){
    if(JSON.stringify(base.setups[camera])!==JSON.stringify(info.setups[camera]))throw Error('Base camera changed');
    const reused=base.frames[frame*3+camera];manifest.frames.push({...reused,file:'../'+reused.file});continue;
   }
   const data=await page.evaluate(([time,index])=>window.renderFrame(time,index),[frame/fps,camera]);
   const file=`frames/camera-${camera}-${String(frame).padStart(4,'0')}.jpg`;
   fs.writeFileSync(path.join(output,file),Buffer.from(data.image.split(',')[1],'base64'));delete data.image;
   manifest.frames.push({...data,file});
   if(camera===2&&frame%15===0)console.log(`Rendered ${frame+1}/${count} instants`);
  }
  fs.writeFileSync(path.join(output,'manifest.json'),JSON.stringify(manifest));
 }finally{if(browser)await browser.close();server.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
