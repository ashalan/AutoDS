function updateValueBubble(value, context) {
  value = value || context.value;
  var $valueBubble = $('.rangeslider__handle', context.$range);
  if ($valueBubble.length) {
    $valueBubble[0].innerHTML = value;
  }
  // $("#calculate").click();
}

$('input[type="range"]').rangeslider({
  polyfill: false,
  onInit: function() {
    updateValueBubble(null, this);
  },
  onSlide: function(position, value) {
    updateValueBubble(value, this);
  }
});
