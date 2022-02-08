module Jekyll
  class CiteTag < Liquid::Tag

    def initialize(tag_name, key, tokens)
      super
      @key = key
    end

    def render(context)
      "<d-cite key='#{@key}'></d-cite>"
    end
  end
end

Liquid::Template.register_tag('cite', Jekyll::CiteTag)
