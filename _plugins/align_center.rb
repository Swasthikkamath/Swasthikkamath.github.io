module Jekyll
  class AlignCenter < Liquid::Block

    def render(context)
      text = super
      site = context.registers[:site]
      converter = site.find_converter_instance(::Jekyll::Converters::Markdown)

      baseurl = context.registers[:site].config['baseurl']
      content = Kramdown::Document.new(text,{remove_span_html_tags:true}).to_html # render markdown in caption
      content = converter.convert(content).gsub(/<\/?p[^>]*>/, "").chomp # remove <p> tags from render output

      "<p style='text-align:center;'>#{content}</p>"
    end

  end
end

Liquid::Template.register_tag('align_center', Jekyll::AlignCenter)

