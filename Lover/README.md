> 我愿意在有限的时间里无限的爱着你。

<style>
/* 头像大小调整 */
.avatar {
    width: 80px;   /* 头像宽度 */
    height: 80px;  /* 头像高度 */
    vertical-align: -20px;
    border-radius: 50%;
    margin-right: 5px;
    margin-bottom: 5px;
    -webkit-box-shadow: 1px 1px 1px rgba(0,0,0,.1), 1px 1px 1px rgba(0,0,0,0.1), 1px 1px 1px rgba(0,0,0,0.1);
    box-shadow: 1px 1px 1px rgba(0,0,0,.1), 1px 1px 1px rgba(0,0,0,0.1), 1px 1px 1px rgba(0,0,0,0.1);
    border: 2px solid #fff;
}
</style>
<body>
<link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/gh/xiaoyanu/file-test@2021.11.24-2/more/lovetime.css" />
  <div style="text-align: center;">
      
    <img class="avatar" src="https://github.com/xinchen-jzx/Notes/blob/main/docs/assets/img/1.jpg" style="width: 100px; height: 100px; border: 2px solid orange; border-radius: 10px;">

    <svg class="my-face" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" width="50" height="50"><path d="M866.944 256.768c-95.488-95.488-250.496-95.488-345.984 0l-13.312 13.312-9.472-9.472c-93.824-93.824-246.656-100.736-343.68-10.368-101.888 94.976-104.064 254.592-6.4 352.256l13.568 13.568 299.264 299.264c25.728 25.728 67.584 25.728 93.44 0l312.576-312.576c95.488-95.488 95.488-250.368 0-345.984zM335.36 352.64c-20.48 0-40.832 6.016-56.704 18.944a85.4912 85.4912 0 0 0-6.912 126.976c9.984 9.984 9.984 26.24 0 36.224l-3.2 3.2c-8.192 8.192-21.632 8.192-29.952 0-52.608-52.608-57.216-138.496-6.528-192.896 26.112-28.032 61.952-43.52 100.096-43.52 14.08 0 25.6 11.52 25.6 25.6v3.072c0 12.416-9.984 22.4-22.4 22.4z" p-id="21617" data-spm-anchor-id="a313x.7781069.0.i46" class="selected" fill="#FF2727"></path></svg>

    <img class="avatar" src="https://github.com/xinchen-jzx/Notes/blob/main/docs/assets/img/2.jpg" style="width: 100px; height: 100px; border: 2px solid orange; border-radius: 10px;">
    
<span id="Timing"></span>
  </div>
  <script>
    function timing() {
      // 开始时间 - 格式: 年-月-日 时:分:秒
      let start = '2025-3-22 00:00:00'
      let startTime = new Date(start).getTime()
      let currentTime = new Date().getTime()
      let difference = currentTime - startTime
      let m =  Math.floor(difference / (1000))
      let mm = m % 60  // 秒
      let f = Math.floor(m / 60)
      let ff = f % 60  // 分钟
      let s = Math.floor(f / 60)  // 小时
      let ss = s % 24
      let day = Math.floor(s / 24 )  // 天数
      return "<br> " + day + " day " + ss + " h " + ff + " m " + mm +' s'
    }
    setInterval(()=>{
      document.getElementById('Timing').innerHTML = timing()
    }, 1000)
  </script>
</body>
