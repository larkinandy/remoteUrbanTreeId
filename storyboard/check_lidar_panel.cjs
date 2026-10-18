const {chromium} = require('C:/Users/larki/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const path = require('path');
const {pathToFileURL} = require('url');
(async () => {
  const browser = await chromium.launch({headless:true,channel:'msedge'});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1000}});
    const errors=[];
    page.on('pageerror', e=>errors.push(e.message));
    await page.goto(pathToFileURL(path.join(__dirname,'story.html')).href);
    await page.locator('[data-source="lidar"]').click();
    const panel=page.locator('.sb-chip-pair');
    await panel.scrollIntoViewIfNeeded();
    const result=await panel.evaluate(el=>({captions:[...el.querySelectorAll('figcaption')].map(e=>e.textContent),images:[...el.querySelectorAll('img')].map(e=>({alt:e.alt,complete:e.complete,width:e.naturalWidth,raw:e.src===e.dataset.rawLidarSrc}))}));
    if(!result.images[0].raw || result.images.some(e=>!e.complete||!e.width)) throw Error(JSON.stringify(result));
    await panel.screenshot({path:path.join(__dirname,'assets/lidar-tab-preview.png')});
    await page.locator('[data-source="aerial"]').click();
    const restored=await panel.evaluate(el=>{const imgs=el.querySelectorAll('img');return imgs[0].src===imgs[0].dataset.aerialSrc&&imgs[1].src===imgs[1].dataset.ndviSrc;});
    await panel.screenshot({path:path.join(__dirname,'assets/aerial-tab-preview.png')});
    if(!restored||errors.length) throw Error(JSON.stringify({restored,errors}));
    console.log(JSON.stringify({lidar:result,aerialRestored:restored,pageErrors:errors}));
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
