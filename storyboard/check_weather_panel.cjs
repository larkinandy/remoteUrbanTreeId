const {chromium}=require('C:/Users/larki/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const path=require('path');
const {pathToFileURL}=require('url');
(async()=>{
 const browser=await chromium.launch({headless:true,channel:'msedge'});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1100}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(pathToFileURL(path.join(__dirname,'story.html')).href);
  await page.locator('[data-source="weather"]').click();
  const weather=page.locator('.sb-weather-example');
  await weather.scrollIntoViewIfNeeded();
  if(!await weather.isVisible()||await page.locator('.sb-crown-examples').isVisible()||await page.locator('.sb-satellite-example').isVisible())throw Error('Incorrect weather tab visibility');
  const loaded=await weather.locator('img').evaluate(e=>e.complete&&e.naturalWidth>0);
  if(!loaded)throw Error('Weather image not loaded');
  await weather.screenshot({path:path.join(__dirname,'assets/weather-tab-preview.png')});
  await page.locator('[data-source="sentinel"]').click();
  if(await weather.isVisible()||!await page.locator('.sb-satellite-example').isVisible())throw Error('Incorrect Sentinel tab visibility');
  await page.setViewportSize({width:390,height:900});
  await page.locator('[data-source="weather"]').click();
  const bounds=await weather.boundingBox();
  if(!bounds||bounds.width>390)throw Error('Weather panel overflows mobile');
  if(errors.length)throw Error(errors.join('\n'));
  console.log('Verified weather image, Sentinel tab restoration, mobile width, and no script errors.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
