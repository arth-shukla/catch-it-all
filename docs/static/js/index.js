// Play clips only while they are in view
document.addEventListener('DOMContentLoaded', function () {
  const videos = document.querySelectorAll('.teaser video, .clip-grid video');
  if (videos.length === 0) return;

  if (!('IntersectionObserver' in window)) {
    videos.forEach(function (video) {
      video.play().catch(function () {});
    });
    return;
  }

  const observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting) {
        entry.target.play().catch(function () {});
      } else {
        entry.target.pause();
      }
    });
  }, { threshold: 0.25 });

  videos.forEach(function (video) {
    observer.observe(video);
  });
});
